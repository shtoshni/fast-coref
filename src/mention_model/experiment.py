import sys
from os import path

import time
import logging
import torch
import json
from collections import OrderedDict

import numpy as np
import pytorch_utils.utils as utils
from mention_model.controller import Controller
from data_utils.utils import load_data
from transformers import get_linear_schedule_with_warmup, AdamW

EPS = 1e-8
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


class Experiment:
    def __init__(self, args, data_dir=None, dataset='litbank',
                 model_dir=None, best_model_dir=None, pretrained_mention_model=None,
                 # Model params
                 seed=0, init_lr=1e-3, fine_tune_lr=1e-5, max_gradient_norm=5.0,
                 max_epochs=20, eval_model=False,
                 # Other params
                 slurm_id=None, **kwargs):

        self.pretrained_mention_model = pretrained_mention_model
        self.slurm_id = slurm_id
        self.max_epochs = max_epochs
        # Set the random seed first
        self.seed = seed

        # Prepare data info
        self.dataset = dataset

        self.data_iter_map = self.load_data(args, training=True)

        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        # Initialize model and training metadata
        self.train_info, self.optimizer, self.optim_scheduler = {}, {}, {}
        self.finetune = (fine_tune_lr is not None)

        self.model = Controller(**kwargs).cuda()
        utils.print_model_info(self.model)

        if eval_model:
            self.load_model(self.best_model_path, model_type='best')
            self.data_iter_map = self.load_data(args, training=False)
            self.final_eval()
        else:
            self.train_info = {'epoch': 0, 'val_perf': 0.0, 'global_steps': 0, 'num_stuck_epochs': 0}
            self.initialize_setup(init_lr, fine_tune_lr)
            self.data_iter_map = self.load_data(args, training=True)
            self.train(max_epochs=max_epochs, max_gradient_norm=max_gradient_norm)
            self.data_iter_map = self.load_data(args, training=False)
            self.final_eval()

    @staticmethod
    def load_data(args, training=True):
        return load_data(args.data_dir, args.max_segment_len, dataset=args.dataset, singleton_file=args.singleton_file,
                         num_train_docs=args.num_train_docs, num_eval_docs=args.num_eval_docs,
                         max_training_segments=args.max_training_segments, num_workers=2, training=training)

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

        num_training_steps = len(self.data_iter_map['train']) * self.max_epochs

        self.optimizer['mem'] = torch.optim.AdamW(
            other_params, lr=init_lr, eps=1e-6, weight_decay=0)
        # self.optimizer_params['mem'] = other_params  # Useful in gradient clipping
        self.optim_scheduler['mem'] = get_linear_schedule_with_warmup(
            self.optimizer['mem'], num_warmup_steps=0, num_training_steps=num_training_steps)
        if fine_tune_lr is not None:
            self.optimizer['doc'] = AdamW(encoder_params, lr=fine_tune_lr, eps=1e-6, weight_decay=1e-2)
            self.optim_scheduler['doc'] = get_linear_schedule_with_warmup(
                self.optimizer['doc'], num_warmup_steps=0, num_training_steps=num_training_steps)

        if not path.exists(self.model_path):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
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
            logger.info("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            model.train()

            for cur_example in self.data_iter_map['train']:
                def handle_example(train_example):
                    loss = model(train_example)[0]
                    total_loss = loss['mention']
                    if total_loss is None:
                        return None

                    if torch.isnan(total_loss):
                        print("Loss is NaN")
                        sys.exit()

                    total_loss.backward()
                    # Perform gradient clipping and update parameters
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_gradient_norm)

                    for key in optimizer:
                        optimizer[key].step()
                        scheduler[key].step()

                    self.train_info['global_steps'] += 1
                    return total_loss.item()

                from copy import deepcopy
                handle_example(deepcopy(cur_example))

                if (self.train_info['global_steps'] + 1) % 50 == 0:
                    logging.info("Steps %d, Max memory %.3f" % (
                        self.train_info['global_steps'] + 1, (torch.cuda.max_memory_allocated() / (1024 ** 3))))
                    torch.cuda.reset_peak_memory_stats()

            # Update epochs done
            self.train_info['epoch'] = epoch + 1
            # Validation performance
            recall = self.eval_model()

            # Update model if validation performance improves
            if recall > self.train_info['val_perf']:
                self.train_info['val_perf'] = recall
                logging.info('Saving best model')
                self.save_model(self.best_model_path)

            # Save model
            self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logger.info("Epoch: %d, Time: %.2f, Recall: %.3f" % (epoch + 1, elapsed_time, recall))

            sys.stdout.flush()

    def eval_model(self, split='dev'):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        dev_examples = self.data_iter_map[split]

        with torch.no_grad():
            log_file = path.join(self.model_dir, split + ".log.jsonl")
            with open(log_file, 'w') as f:
                total_recall = 0
                total_gold = 0.0
                all_golds = 0.0
                # Output file to write the outputs
                for dev_example in dev_examples:
                    pred_mentions, mention_scores, recall, filtered_gold = model(dev_example)

                    log_example = dict(dev_example)
                    log_example["pred_mentions"] = pred_mentions
                    log_example["mention_scores"] = mention_scores

                    all_golds += sum([len(cluster) for cluster in dev_example["clusters"]])
                    total_recall += recall
                    total_gold += filtered_gold

                    del log_example["padded_sent"]
                    del log_example["sent_len_list"]

                    f.write(json.dumps(log_example) + "\n")

                logger.info(log_file)

        logger.info(f"Identified mentions: {total_recall}, Total mentions: {all_golds}, "
                    f"Filtered mentions: {total_gold}\n\n")
        logger.info("Recall: %.3f" % (total_recall/total_gold))

        return total_recall/total_gold

    def final_eval(self):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        self.load_model(self.best_model_path)
        logging.info("Loading best model after epoch: %d" %
                     self.train_info['epoch'])

        perf_file = path.join(self.model_dir, "perf.txt")
        with open(perf_file, 'w') as f:
            for split in ['Train', 'Dev', 'Test']:
                logging.info('\n')
                logging.info('%s' % split)
                split_f1, _ = self.eval_model(split.lower())
                logging.info('Calculated F1: %.3f' % split_f1)

                f.write("%s\t%.4f\n" % (split, split_f1))
            logging.info("Final performance summary at %s" % perf_file)

        sys.stdout.flush()

    def load_model(self, location, model_type='last'):
        checkpoint = torch.load(location, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        if model_type != 'best':
            param_groups = ['mem', 'doc'] if self.finetune else ['mem']
            for param_group in param_groups:
                self.optimizer[param_group].load_state_dict(
                    checkpoint['optimizer'][param_group])
                self.optim_scheduler[param_group].load_state_dict(
                    checkpoint['scheduler'][param_group])
        self.train_info = checkpoint['train_info']
        torch.set_rng_state(checkpoint['rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])

    def save_model(self, location, model_type='last'):
        """Save model"""
        model_state_dict = OrderedDict(self.model.state_dict())
        if not self.finetune:
            for key in self.model.state_dict():
                if 'lm_encoder.' in key:
                    del model_state_dict[key]
        save_dict = {
            'train_info': self.train_info,
            'model': model_state_dict,
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'optimizer': {},
            'scheduler': {},
        }
        if model_type != 'best':
            param_groups = ['mem', 'doc'] if self.finetune else ['mem']
            for param_group in param_groups:
                save_dict['optimizer'][param_group] = self.optimizer[param_group].state_dict()
                save_dict['scheduler'][param_group] = self.optim_scheduler[param_group].state_dict()

        torch.save(save_dict, location)
        logging.info(f"Model saved at: {location}")

