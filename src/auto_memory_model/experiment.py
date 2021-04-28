import sys
from os import path
import os
import time
import logging
import torch
import json
from collections import OrderedDict
import numpy as np
import random
from transformers import get_linear_schedule_with_warmup, AdamW

from auto_memory_model.utils import action_sequences_to_clusters
from data_utils.utils import load_data
from coref_utils.conll import evaluate_conll
from coref_utils.utils import get_mention_to_cluster
from coref_utils.metrics import CorefEvaluator
import pytorch_utils.utils as utils
from auto_memory_model.controller.utils import pick_controller
from data_utils.tensorize_dataset import TensorizeDataset

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


class Experiment:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cluster threshold is used to determine the minimum size of clusters for metric calculation
        self.cluster_threshold = 1 if self.train_with_singletons else 2
        if self.dataset == 'litbank':
            self.update_frequency = 10  # Frequency in terms of # of documents after which logs are printed
            self.canonical_cluster_threshold = 1
        elif self.dataset == 'ontonotes':
            # OntoNotes
            self.update_frequency = 500
            self.canonical_cluster_threshold = 2
        elif self.dataset == 'preco':
            # OntoNotes
            self.update_frequency = 100
            self.canonical_cluster_threshold = 1

        self.orig_data_map = self.load_data()
        self.data_processor = TensorizeDataset()
        self.data_iter_map = {}

        self.model = None
        self.finetune = (self.fine_tune_lr is not None)
        self.model_args = kwargs

        self.model_path = path.join(self.model_dir, 'model.pth')
        self.best_model_path = path.join(self.best_model_dir, 'model.pth')

        self.optimizer, self.optim_scheduler, self.scaler = {}, {}, None

        if not self.eval_model:
            # Train info is a dictionary to keep around important training variables
            self.train_info = {'epoch': 0, 'val_perf': 0.0, 'global_steps': 0, 'num_stuck_evals': 0}
            self.patience = 10  # Maximum evals without improvement in dev performance
            # Initialize model and training metadata
            self.setup_training()
            self.train()

            self.load_model(self.best_model_path, model_type='best')
            logger.info("Loading best model after epoch: %d" % self.train_info['epoch'])
        else:
            self.setup_eval()

        self.perform_final_eval()

    def load_data(self):
        return load_data(self.data_dir, self.max_segment_len, dataset=self.dataset, singleton_file=self.singleton_file,
                         num_train_docs=self.num_train_docs, num_eval_docs=self.num_eval_docs)

    def setup_training(self):
        self.model = pick_controller(
            device=self.device, finetune=self.finetune, **self.model_args).to(self.device)

        self.initialize_optimizers()

        if path.exists(self.model_path):
            logging.info('Loading previous model: %s' % self.model_path)
            self.load_model(self.model_path)

        utils.print_model_info(self.model)
        sys.stdout.flush()

        self.data_iter_map['dev'] = self.data_processor.tensorize_data(self.orig_data_map['dev'])
        self.data_iter_map['train'] = self.data_processor.tensorize_data(self.orig_data_map['train'], training=True)

    def setup_eval(self):
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model = pick_controller(device=self.device, **checkpoint['model_args']).to(self.device)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        print(checkpoint['model_args'])
        # Finally evaluate model
        if self.eval_max_ents is not None:
            self.model.set_max_ents(self.eval_max_ents)
        if self.use_gold_ments is not None:
            self.model.use_gold_ments = self.use_gold_ments

        if self.dataset != self.model.dataset:
            # Change the default mention detection constants
            if self.max_span_width is not None:
                self.model.max_span_width = self.max_span_width
            if self.top_span_ratio is not None:
                self.model.top_span_ratio = self.top_span_ratio

    def initialize_optimizers(self):
        """Initialize model + optimizer(s). Check if there's a checkpoint in which case we resume from there."""
        torch.random.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.scaler = torch.cuda.amp.GradScaler()

        num_training_steps = len(self.orig_data_map['train']) * self.max_epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        self.optimizer['mem'] = torch.optim.Adam(self.model.get_params()[1], lr=self.init_lr, eps=1e-6)
        # self.optimizer['mem'] = FusedAdam(self.model.get_params()[1], lr=self.init_lr)
        # No warmup for model params
        self.optim_scheduler['mem'] = get_linear_schedule_with_warmup(
            self.optimizer['mem'], num_warmup_steps=0, num_training_steps=num_training_steps)

        if self.fine_tune_lr is not None:
            no_decay = ['bias', 'LayerNorm.weight']
            encoder_params = self.model.get_params(named=True)[0]
            grouped_param = [
                {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
                 'lr': self.fine_tune_lr,
                 'weight_decay': 1e-2},
                {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
                 'lr': self.fine_tune_lr,
                 'weight_decay': 0.0}
            ]

            # self.optimizer['doc'] = FusedAdam(grouped_param, lr=self.init_lr, weight_decay=0.01)
            self.optimizer['doc'] = AdamW(grouped_param, lr=self.fine_tune_lr, eps=1e-6)
            self.optim_scheduler['doc'] = get_linear_schedule_with_warmup(
                self.optimizer['doc'], num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    def train(self):
        """Train model"""
        if self.train_info['num_stuck_evals'] >= self.patience:
            return

        model, optimizer, scheduler, scaler = self.model, self.optimizer, self.optim_scheduler, self.scaler
        # Setup training
        model.train()

        for epoch in range(self.train_info['epoch'], self.max_epochs):
            logger.info("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()

            train_data = self.data_iter_map['train']
            np.random.shuffle(train_data)

            encoder_params, task_params = model.get_params()

            for cur_example in train_data:
                def handle_example(example):
                    self.train_info['global_steps'] += 1
                    for key in optimizer:
                        optimizer[key].zero_grad()

                    with torch.cuda.amp.autocast():
                        loss = model.forward_training(example)
                        total_loss = loss['total']
                        if total_loss is None:
                            return None

                    scaler.scale(total_loss).backward()
                    for key in optimizer:
                        scaler.unscale_(optimizer[key])

                    torch.nn.utils.clip_grad_norm_(encoder_params, self.max_gradient_norm)
                    torch.nn.utils.clip_grad_norm_(task_params, self.max_gradient_norm)

                    for key in optimizer:
                        scaler.step(optimizer[key])
                        scheduler[key].step()

                    scaler.update()
                    return total_loss.item()

                example_loss = handle_example(cur_example)

                if self.eval_per_k_steps and (self.train_info['global_steps'] % self.eval_per_k_steps == 0):
                    fscore = self.periodic_model_eval()
                    # Get elapsed time
                    elapsed_time = time.time() - start_time
                    logger.info("Steps: %d, F1: %.1f, Max F1: %.1f, Time: %.2f"
                                % (self.train_info['global_steps'], fscore, self.train_info['val_perf'], elapsed_time))

                    if self.train_info['num_stuck_evals'] >= self.patience:
                        return

                if self.train_info['global_steps'] % self.update_frequency == 0:
                    logger.info('{} {:.3f} Max mem {:.3f} GB'.format(
                        cur_example["doc_key"], example_loss,
                        (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0)
                    )
                    torch.cuda.reset_peak_memory_stats()

            sys.stdout.flush()
            # Update epochs done
            self.train_info['epoch'] = epoch + 1
            if not self.eval_per_k_steps:
                fscore = self.periodic_model_eval()
                # Get elapsed time
                elapsed_time = time.time() - start_time
                logger.info("Epoch: %d, F1: %.1f, Max F1: %.1f, Time: %.2f"
                            % (epoch + 1, fscore, self.train_info['val_perf'], elapsed_time))

            sys.stdout.flush()
            logger.handlers[0].flush()

    def periodic_model_eval(self):
        # Dev performance
        cluster_threshold = max(self.cluster_threshold, self.canonical_cluster_threshold)
        fscore = self.evaluate_model(cluster_threshold=cluster_threshold)['fscore']

        # Assume that the model didn't improve
        self.train_info['num_stuck_evals'] += 1

        # Update model if dev performance improves
        if fscore > self.train_info['val_perf']:
            self.train_info['num_stuck_evals'] = 0
            self.train_info['val_perf'] = fscore
            logger.info('Saving best model')
            self.save_model(self.best_model_path, model_type='best')

        # Save model
        if self.to_save_model:
            if self.dataset == 'litbank':
                # Can train LitBank in one slurm job - Save Disk I/o time
                if (self.train_info['epoch']) % 10 == 0:
                    self.save_model(self.model_path)
            else:
                self.save_model(self.model_path)

        # Go back to training mode
        self.model.train()
        return fscore

    def evaluate_model(self, split='dev', final_eval=False, cluster_threshold=1):
        """Eval model"""
        model = self.model
        model.eval()

        num_gt_clusters, num_pred_clusters = 0, 0
        inference_time = 0.0

        with torch.no_grad():
            log_file = path.join(self.model_dir, split + ".log.jsonl")
            with open(log_file, 'w') as f:
                # Capture the auxiliary action accuracy
                corr_actions, total_actions = 0.0, 0.0
                oracle_evaluator, evaluator = CorefEvaluator(), CorefEvaluator()
                coref_predictions, subtoken_maps = {}, {}

                logger.info(f"Evaluating on {len(self.data_iter_map[split])} examples")
                for example in self.data_iter_map[split]:
                    start_time = time.time()
                    action_list, pred_mentions, gt_actions = model(example)
                    # Predicted cluster
                    raw_predicted_clusters = action_sequences_to_clusters(action_list, pred_mentions)
                    predicted_clusters, mention_to_predicted =\
                        get_mention_to_cluster(raw_predicted_clusters, threshold=cluster_threshold)
                    gold_clusters, mention_to_gold =\
                        get_mention_to_cluster(example["clusters"], threshold=cluster_threshold)
                    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

                    elapsed_time = time.time() - start_time
                    inference_time += elapsed_time

                    coref_predictions[example["doc_key"]] = predicted_clusters
                    subtoken_maps[example["doc_key"]] = example["subtoken_map"]

                    total_actions += len(action_list)
                    # Update the number of clusters
                    num_gt_clusters += len(gold_clusters)
                    num_pred_clusters += len(predicted_clusters)

                    # Oracle clustering
                    oracle_clusters = action_sequences_to_clusters(gt_actions, pred_mentions)
                    oracle_clusters, mention_to_oracle = \
                        get_mention_to_cluster(oracle_clusters, threshold=cluster_threshold)
                    oracle_evaluator.update(oracle_clusters, gold_clusters, mention_to_oracle, mention_to_gold)

                    log_example = dict(example)
                    log_example["pred_mentions"] = pred_mentions
                    if cluster_threshold != 1:
                        # For cluster threshold 1, raw and processed clusters are one and the same
                        log_example["raw_predicted_clusters"] = raw_predicted_clusters
                    log_example["pred_actions"] = action_list
                    log_example["predicted_clusters"] = predicted_clusters

                    del log_example["tensorized_sent"]
                    for key in list(log_example.keys()):
                        if isinstance(log_example[key], torch.Tensor):
                            del log_example[key]

                    f.write(json.dumps(log_example) + "\n")

                # Print individual metrics
                result_dict = OrderedDict()
                indv_metrics_list = ['MUC', 'Bcub', 'CEAFE']
                perf_str = ""
                for indv_metric, indv_evaluator in zip(indv_metrics_list, evaluator.evaluators):
                    perf_str += ", " + indv_metric + ": {:.1f}".format(indv_evaluator.get_f1() * 100)
                    result_dict[indv_metric] = OrderedDict()
                    result_dict[indv_metric]['recall'] = round(indv_evaluator.get_recall() * 100, 1)
                    result_dict[indv_metric]['precision'] = round(indv_evaluator.get_precision() * 100, 1)
                    result_dict[indv_metric]['fscore'] = round(indv_evaluator.get_f1() * 100, 1)

                fscore = evaluator.get_f1() * 100
                result_dict['fscore'] = round(fscore, 1)
                logger.info("F-score: %.1f %s" % (fscore, perf_str))

                # (1) Only use CoNLL evaluator script for final evaluation
                # (2) CoNLL score only makes sense when the evaluation is using the canonical cluster threshold
                use_conll = (cluster_threshold == self.canonical_cluster_threshold)
                # (3) Check if the scorer and CoNLL annotation directory exist
                path_exists_bool = False
                if self.conll_scorer is not None and self.conll_data_dir is not None:
                    path_exists_bool = path.exists(self.conll_scorer) and path.exists(self.conll_data_dir)

                try:
                    if final_eval and use_conll and path_exists_bool:
                        gold_path = path.join(self.conll_data_dir, f'{split}.conll')
                        prediction_file = path.join(self.model_dir, f'{split}.conll')
                        conll_results = evaluate_conll(
                            self.conll_scorer, gold_path, coref_predictions, subtoken_maps, prediction_file)

                        for indv_metric in indv_metrics_list:
                            result_dict[indv_metric] = OrderedDict()
                            result_dict[indv_metric]['recall'] = round(conll_results[indv_metric.lower()]["r"], 1)
                            result_dict[indv_metric]['precision'] = round(conll_results[indv_metric.lower()]["p"], 1)
                            result_dict[indv_metric]['fscore'] = round(conll_results[indv_metric.lower()]["f"], 1)

                        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
                        result_dict['fscore'] = round(average_f1, 1)

                        logger.info("(CoNLL) F-score : %.1f, MUC: %.1f, Bcub: %.1f, CEAFE: %.1f"
                                    % (average_f1, conll_results["muc"]["f"], conll_results['bcub']["f"],
                                        conll_results['ceafe']["f"]))
                except AttributeError:
                    pass

                logger.info("Oracle F-score: %.3f" % oracle_evaluator.get_prf()[2])
                logger.info(log_file)
                logger.handlers[0].flush()

        logger.info("Inference time: %.2f" % inference_time)

        return result_dict

    def perform_final_eval(self):
        """Evaluate the model on train, dev, and test"""
        perf_file = path.join(self.model_dir, "perf.json")
        if self.slurm_id:
            parent_dir = path.dirname(path.normpath(self.model_dir))
            perf_dir = path.join(parent_dir, "perf")
            if not path.exists(perf_dir):
                os.makedirs(perf_dir)
            perf_file = path.join(perf_dir, self.slurm_id + ".json")

        output_dict = {'model_dir': self.model_dir}
        for key, val in self.model_args.items():
            output_dict[key] = val

        # for split in ['test', 'dev', 'train']:
        for split in ['test', 'dev']:
            if split == 'train':
                # We had truncated the document earlier - But now we want to process the whole training document
                self.data_iter_map['train'] = self.data_processor.tensorize_data(self.orig_data_map['train'])
            if split not in self.data_iter_map:
                self.data_iter_map[split] = self.data_processor.tensorize_data(self.orig_data_map[split])

            cluster_thresholds = [self.canonical_cluster_threshold]
            for cluster_threshold in cluster_thresholds:
                logging.info('\n')
                logging.info('%s' % split.capitalize())
                result_dict = self.evaluate_model(split, final_eval=True, cluster_threshold=cluster_threshold)
                if split != 'test':
                    logging.info('Calculated F1: %.3f' % result_dict['fscore'])

                output_dict[f"{split}_{cluster_threshold}"] = result_dict
                if cluster_threshold == self.canonical_cluster_threshold:
                    output_dict[f"{split}"] = result_dict

        json.dump(output_dict, open(perf_file, 'w'), indent=2)

        logging.info("Final performance summary at %s" % perf_file)
        sys.stdout.flush()

    def load_model(self, location, model_type='last'):
        checkpoint = torch.load(location, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        if model_type != 'best':
            for param_group in checkpoint['optimizer']:
                self.optimizer[param_group].load_state_dict(
                    checkpoint['optimizer'][param_group])
                self.optim_scheduler[param_group].load_state_dict(
                    checkpoint['scheduler'][param_group])

            if 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])

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
            'scaler': self.scaler.state_dict(),
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'optimizer': {},
            'scheduler': {},
            'model_args': self.model_args,
        }
        if model_type != 'best':
            param_groups = ['mem', 'doc'] if self.finetune else ['mem']
            for param_group in param_groups:
                save_dict['optimizer'][param_group] = self.optimizer[param_group].state_dict()
                save_dict['scheduler'][param_group] = self.optim_scheduler[param_group].state_dict()

        torch.save(save_dict, location)
        logging.info(f"Model saved at: {location}")
