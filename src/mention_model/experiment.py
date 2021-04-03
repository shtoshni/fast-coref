import sys
from os import path

import time
import logging
import torch
from collections import defaultdict, OrderedDict

import numpy as np
import pytorch_utils.utils as utils
from mention_model.controller import Controller
from data_utils.utils import load_data
from coref_utils.utils import remove_singletons
from transformers import get_linear_schedule_with_warmup, AdamW

EPS = 1e-8
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Experiment:
    def __init__(self, data_dir=None, dataset='litbank',
                 model_dir=None, best_model_dir=None, pretrained_model=None,
                 # Model params
                 seed=0, init_lr=1e-3, max_gradient_norm=5.0,
                 max_epochs=20, max_segment_len=128, eval=False,
                 num_train_docs=None,
                 # Other params
                 slurm_id=None, train_with_singletons=False,
                 **kwargs):

        self.pretrained_model = pretrained_model
        self.slurm_id = slurm_id
        # Set the random seed first
        self.seed = seed
        # Prepare data info
        self.train_examples, self.dev_examples, self.test_examples \
            = load_data(data_dir, max_segment_len, dataset=dataset)
        # self.dev_examples = self.dev_examples[:20]
        if num_train_docs is not None:
            self.train_examples = self.train_examples[:num_train_docs]

        if train_with_singletons is False:
            print("Removing singletons")
            self.train_examples, self.dev_examples, self.test_examples = \
                [remove_singletons(x) for x in [self.train_examples, self.dev_examples, self.test_examples]]

        self.data_iter_map = {"train": self.train_examples,
                              "valid": self.dev_examples,
                              "test": self.test_examples}
        self.max_epochs = max_epochs

        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        # Initialize model and training metadata
        self.model = Controller(**kwargs)
        self.model = self.model.cuda()

        self.train_info, self.optimizer, self.optim_scheduler = {}, {}, {}

        self.initialize_setup(init_lr=init_lr)
        self.model = self.model.cuda()
        utils.print_model_info(self.model)

        if not eval:
            if self.pretrained_model is not None:
                model_state_dict = torch.load(self.pretrained_model)
                print(model_state_dict.keys())
                self.model.load_state_dict(model_state_dict, strict=False)
                self.eval_model(split='valid')
                # self.eval_model(split='test')
                sys.exit()
            else:
                self.train(max_epochs=max_epochs,
                           max_gradient_norm=max_gradient_norm)
        # Finally evaluate model
        self.final_eval(model_dir)

    def initialize_setup(self, init_lr, fine_tune_lr):
        """Initialize model + optimizer(s). Check if there's a checkpoint in which case we resume from there."""
        other_params = []
        encoder_params = []
        encoder_param_names = set([name for name, _ in self.model.named_parameters() if "lm_encoder." in name])

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in encoder_param_names:
                    encoder_params.append(param)
                    continue
                else:
                    other_params.append(param)

        num_training_steps = len(self.train_examples) * self.max_epochs

        self.optimizer['mem'] = torch.optim.AdamW(
            other_params, lr=init_lr, eps=1e-6, weight_decay=1e-2)
        # self.optimizer_params['mem'] = other_params  # Useful in gradient clipping
        self.optim_scheduler['mem'] = get_linear_schedule_with_warmup(
            self.optimizer['mem'], num_warmup_steps=0, num_training_steps=num_training_steps)
        if fine_tune_lr is not None:
            self.optimizer['doc'] = AdamW(encoder_params, lr=fine_tune_lr, eps=1e-6, weight_decay=0.0)
            self.optim_scheduler['doc'] = get_linear_schedule_with_warmup(
                self.optimizer['doc'], num_warmup_steps=int(0.1 * num_training_steps),
                num_training_steps=num_training_steps)

        if not path.exists(self.model_path):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            # Try to initialize the mention model part
        else:
            logging.info('Loading previous model: %s' % self.model_path)
            # Load model
            self.load_model(self.model_path)

    def train(self, max_epochs, max_gradient_norm):
        """Train model"""
        model = self.model
        epochs_done = self.train_info['epoch']
        optimizer = self.optimizer
        scheduler = self.optim_scheduler

        for epoch in range(epochs_done, max_epochs):
            print("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            model.train()
            np.random.shuffle(self.train_examples)

            for idx, cur_example in enumerate(self.train_examples):
                def handle_example(train_example):
                    self.train_info['global_steps'] += 1
                    loss = model(train_example)
                    total_loss = loss['mention']

                    if torch.isnan(total_loss):
                        print("Loss is NaN")
                        sys.exit()
                    # Backprop
                    optimizer.zero_grad()
                    total_loss.backward()
                    # Perform gradient clipping and update parameters
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_gradient_norm)

                    optimizer.step()

                from copy import deepcopy
                handle_example(deepcopy(cur_example))

                if (idx + 1) % 50 == 0:
                    print("Steps %d, Max memory %.3f" % (idx + 1, (torch.cuda.max_memory_allocated() / (1024 ** 3))))
                    torch.cuda.reset_peak_memory_stats()
                    # print("Current memory %.3f" % (torch.cuda.memory_allocated() / (1024 ** 3)))
                    # print(torch.cuda.memory_summary())

            # Update epochs done
            self.train_info['epoch'] = epoch + 1
            # Validation performance
            fscore, threshold = self.eval_model()

            scheduler.step(fscore)

            # Update model if validation performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['val_perf'] = fscore
                self.train_info['threshold'] = threshold
                logging.info('Saving best model')
                self.save_model(self.best_model_path)

            # Save model
            self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logging.info("Epoch: %d, Time: %.2f, F-score: %.3f"
                         % (epoch + 1, elapsed_time, fscore))

            sys.stdout.flush()

    def eval_preds(self, pred_mention_probs, gold_mentions, threshold=0.5):
        pred_mentions = (pred_mention_probs >= threshold).float()
        total_corr = torch.sum(pred_mentions * gold_mentions)

        return total_corr, torch.sum(pred_mentions), torch.sum(gold_mentions)

    def eval_model(self, split='valid', threshold=None):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        dev_examples = self.data_iter_map[split]

        with torch.no_grad():
            total_recall = 0
            total_gold = 0.0
            all_golds = 0.0
            # Output file to write the outputs
            agg_results = {}
            for dev_example in dev_examples:
                preds, y, cand_starts, cand_ends, recall = model(dev_example)

                all_golds += sum([len(cluster) for cluster in dev_example["clusters"]])
                total_gold += torch.sum(y).item()
                total_recall += recall

                if threshold:
                    corr, total_preds, total_y = self.eval_preds(
                        preds, y, threshold=threshold)
                    if threshold not in agg_results:
                        agg_results[threshold] = defaultdict(float)

                    x = agg_results[threshold]
                    x['corr'] += corr
                    x['total_preds'] += total_preds
                    x['total_y'] += total_y
                    prec = x['corr']/(x['total_preds'] + EPS)
                    recall = x['corr']/x['total_y']
                    x['fscore'] = 2 * prec * recall/(prec + recall + EPS)
                else:
                    threshold_range = np.arange(0.0, 0.5, 0.01)
                    for cur_threshold in threshold_range:
                        corr, total_preds, total_y = self.eval_preds(
                            preds, y, threshold=cur_threshold)
                        if cur_threshold not in agg_results:
                            agg_results[cur_threshold] = defaultdict(float)

                        x = agg_results[cur_threshold]
                        x['corr'] += corr
                        x['total_preds'] += total_preds
                        x['total_y'] += total_y
                        prec = x['corr']/x['total_preds']
                        recall = x['corr']/x['total_y']
                        x['fscore'] = 2 * prec * recall/(prec + recall + EPS)

        if threshold:
            max_fscore = agg_results[threshold]['fscore']
        else:
            max_fscore, threshold = 0, 0.0
            for key in agg_results:
                if agg_results[key]['fscore'] > max_fscore:
                    max_fscore = agg_results[key]['fscore']
                    threshold = key

            logging.info("Max F-score: %.3f, Threshold: %.3f" %
                         (max_fscore, threshold))
        print(total_recall, total_gold)
        print(total_recall, all_golds)
        logging.info("Recall: %.3f" % (total_recall/total_gold))
        return max_fscore, threshold

    def final_eval(self, model_dir):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        self.load_model(self.best_model_path)
        logging.info("Loading best model after epoch: %d" %
                     self.train_info['epoch'])
        logging.info("Threshold: %.3f" % self.train_info['threshold'])
        threshold = self.train_info['threshold']

        perf_file = path.join(self.model_dir, "perf.txt")
        with open(perf_file, 'w') as f:
            for split in ['Train', 'Valid', 'Test']:
                logging.info('\n')
                logging.info('%s' % split)
                split_f1, _ = self.eval_model(
                    split.lower(), threshold=threshold)
                logging.info('Calculated F1: %.3f' % split_f1)

                f.write("%s\t%.4f\n" % (split, split_f1))
            logging.info("Final performance summary at %s" % perf_file)

        sys.stdout.flush()

    def load_model(self, location):
        checkpoint = torch.load(location)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.optimizer.load_state_dict(
            checkpoint['optimizer'])
        self.optim_scheduler.load_state_dict(
            checkpoint['scheduler'])
        self.train_info = checkpoint['train_info']
        torch.set_rng_state(checkpoint['rng_state'])

    def save_model(self, location):
        """Save model"""
        model_state_dict = OrderedDict(self.model.state_dict())
        for key in self.model.state_dict():
            if 'bert.' in key:
                del model_state_dict[key]
        torch.save({
            'train_info': self.train_info,
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.optim_scheduler.state_dict(),
            'rng_state': torch.get_rng_state(),
        }, location)
        logging.info("Model saved at: %s" % (location))
