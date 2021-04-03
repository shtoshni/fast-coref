import sys
from os import path
import os
import time
import logging
import torch
import json
from collections import defaultdict, OrderedDict
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW

from auto_memory_model.utils import action_sequences_to_clusters
from data_utils.utils import load_data
from coref_utils.conll import evaluate_conll
from coref_utils.utils import get_mention_to_cluster
from coref_utils.metrics import CorefEvaluator
import pytorch_utils.utils as utils
from coref_utils.utils import remove_singletons
from auto_memory_model.controller.utils import pick_controller
import wandb

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


class Experiment:
    def __init__(self, args, data_dir=None, dataset='litbank',
                 model_dir=None, best_model_dir=None, singleton_file=None,
                 pretrained_mention_model=None,
                 # Model params
                 seed=0, init_lr=3e-4, fine_tune_lr=None, max_gradient_norm=10.0,
                 max_epochs=20, max_segment_len=512, eval_model=False,
                 mem_type="unbounded", train_with_singletons=False,
                 eval_max_ents=None, use_gold_ments=False,
                 # Other params
                 slurm_id=None, conll_data_dir=None, conll_scorer=None, **kwargs):
        self.args = args

        # Set the random seed first
        self.seed = seed
        self.model_args = vars(args)
        self.pretrained_mention_model = pretrained_mention_model

        # Cluster threshold is used to determine the minimum size of clusters for metric calculation
        self.dataset = dataset
        self.train_examples, self.dev_examples, self.test_examples \
            = load_data(data_dir, max_segment_len, dataset=self.dataset, singleton_file=singleton_file,
                        num_train_docs=args.num_train_docs, num_eval_docs=args.num_eval_docs,
                        max_training_segments=args.max_training_segments, num_workers=2)

        self.finetune = (fine_tune_lr is not None)
        self.train_with_singletons = train_with_singletons

        if train_with_singletons:
            self.cluster_threshold = 1
        else:
            self.cluster_threshold = 2
            # Remove singletons from training set
            # self.train_examples = remove_singletons(self.train_examples)

        self.canonical_cluster_threshold = 1
        if self.dataset == 'litbank':
            self.update_frequency = 10  # Frequency in terms of # of documents after which logs are printed
            self.max_stuck_epochs = 10  # Maximum epochs without improvement in dev performance
            self.canonical_cluster_threshold = 1
        else:
            # OntoNotes
            self.update_frequency = 100
            self.max_stuck_epochs = 10
            self.canonical_cluster_threshold = 2

        self.data_iter_map = {"train": self.train_examples, "dev": self.dev_examples, "test": self.test_examples}

        self.max_epochs = max_epochs
        self.slurm_id = slurm_id  # Useful to keep this around for grid searches

        # CoNLL scorer and data in CoNLL format. Not a requirement as the python script gets pretty much
        # the same numbers.
        self.conll_scorer = conll_scorer
        self.conll_data_dir = conll_data_dir

        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_info, self.optimizer, self.optim_scheduler = {}, {}, {}
        run_name = str(("slurm " if slurm_id is not None else "") + path.basename(model_dir))
        wandb.init(name=run_name, project='Coreference',
                   notes='Training with Longformer', resume=True,
                   dir=args.base_model_dir)

        if eval_model:
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model = pick_controller(device=self.device, **checkpoint['model_args']).to(self.device)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            print(checkpoint['model_args'])
            # Finally evaluate model
            if eval_max_ents is not None:
                self.model.set_max_ents(eval_max_ents)
            if use_gold_ments is not None:
                self.model.use_gold_ments = use_gold_ments

            if args.dataset != self.model.dataset:
                # Change the default mention detection constants
                if args.max_span_width is not None:
                    self.model.max_span_width = args.max_span_width
                if args.top_span_ratio is not None:
                    self.model.top_span_ratio = args.top_span_ratio

            self.final_eval()
        else:
            # Initialize model and training metadata
            self.model = pick_controller(
                mem_type=mem_type, dataset=dataset, device=self.device,
                finetune=self.finetune, **kwargs).to(self.device)

            # Train info is a dictionary to keep around important training variables
            self.train_info = {'epoch': 0, 'val_perf': 0.0, 'global_steps': 0, 'num_stuck_epochs': 0}
            self.initialize_setup(init_lr=init_lr, fine_tune_lr=fine_tune_lr)
            utils.print_model_info(self.model)
            sys.stdout.flush()

            self.train(max_gradient_norm=max_gradient_norm)

            self.load_model(self.best_model_path, model_type='best')
            logger.info("Loading best model after epoch: %d" % self.train_info['epoch'])
            self.final_eval()

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
            # Try to initialize the mention model part
            if path.exists(self.pretrained_mention_model):
                print("Found pretrained model!!")
                checkpoint = torch.load(self.pretrained_mention_model)
                self.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            logging.info('Loading previous model: %s' % self.model_path)
            # Load model
            self.load_model(self.model_path)

    def train(self, max_gradient_norm):
        """Train model"""
        model = self.model
        epochs_done = self.train_info['epoch']
        optimizer = self.optimizer
        scheduler = self.optim_scheduler

        if self.train_info['num_stuck_epochs'] >= self.max_stuck_epochs:
            return

        for epoch in range(epochs_done, self.max_epochs):
            logger.info("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            # Setup training
            model.train()

            for cur_example in self.train_examples:
                def handle_example(example):
                    # Backprop
                    for key in optimizer:
                        optimizer[key].zero_grad()

                    # Send the copy of the example, as the document could be truncated during training
                    from copy import deepcopy
                    loss = model(example)[0]
                    total_loss = loss['total']
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

                example_loss = handle_example(cur_example)

                if self.train_info['global_steps'] % self.update_frequency == 0:
                    logger.info('{} {:.3f} Max mem {:.3f} GB'.format(
                        cur_example["doc_key"], example_loss,
                        (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0)
                    )
                    wandb.log({"Epoch": epoch,
                               "Step": self.train_info['global_steps'],
                               "Train Loss": example_loss})
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except AttributeError:
                            # In case of an earlier torch version
                            torch.cuda.reset_max_memory_allocated()

            sys.stdout.flush()
            # Update epochs done
            self.train_info['epoch'] = epoch + 1

            # Dev performance
            cluster_threshold = max(self.cluster_threshold, self.canonical_cluster_threshold)
            # print("Cluster threshold:", cluster_threshold)
            fscore = self.eval_model(cluster_threshold=cluster_threshold)['fscore']

            # Assume that the model didn't improve
            self.train_info['num_stuck_epochs'] += 1

            # Update model if dev performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['num_stuck_epochs'] = 0
                self.train_info['val_perf'] = fscore
                logger.info('Saving best model')
                self.save_model(self.best_model_path, model_type='best')

            # Save model
            if self.dataset == 'litbank':
                # Can train LitBank in one slurm job - Save Disk I/o time
                if epoch == self.max_epochs - 1:
                    self.save_model(self.model_path)
            else:
                self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logger.info("Epoch: %d, F1: %.1f, Max F1: %.1f, Time: %.2f"
                        % (epoch + 1, fscore, self.train_info['val_perf'], elapsed_time))

            wandb.log({"Epoch": epoch, "Val F1": fscore})

            sys.stdout.flush()
            logger.handlers[0].flush()

            if self.train_info['num_stuck_epochs'] >= self.max_stuck_epochs:
                return

    def eval_model(self, split='dev', final_eval=False, cluster_threshold=1):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        data_iter = self.data_iter_map[split]

        pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)
        num_gt_clusters, num_pred_clusters = 0, 0

        inference_time = 0.0

        with torch.no_grad():
            log_file = path.join(self.model_dir, split + ".log.jsonl")
            with open(log_file, 'w') as f:
                # Capture the auxiliary action accuracy
                corr_actions = 0.0
                total_actions = 0.0

                # Output file to write the outputs
                evaluator = CorefEvaluator()
                oracle_evaluator = CorefEvaluator()
                coref_predictions, subtoken_maps = {}, {}

                for example in data_iter:
                    start_time = time.time()
                    loss, action_list, pred_mentions, mention_scores, gt_actions = model(example)

                    for pred_action, gt_action in zip(action_list, gt_actions):
                        pred_class_counter[pred_action[1]] += 1
                        gt_class_counter[gt_action[1]] += 1

                        if tuple(pred_action) == tuple(gt_action):
                            corr_actions += 1

                    total_actions += len(action_list)

                    predicted_clusters = action_sequences_to_clusters(action_list, pred_mentions)
                    elapsed_time = time.time() - start_time
                    inference_time += elapsed_time

                    log_example = dict(example)
                    log_example["pred_mentions"] = pred_mentions
                    log_example["mention_scores"] = mention_scores
                    log_example["raw_predicted_clusters"] = predicted_clusters

                    predicted_clusters, mention_to_predicted =\
                        get_mention_to_cluster(predicted_clusters, threshold=cluster_threshold)
                    gold_clusters, mention_to_gold =\
                        get_mention_to_cluster(example["clusters"], threshold=cluster_threshold)

                    coref_predictions[example["doc_key"]] = predicted_clusters
                    subtoken_maps[example["doc_key"]] = example["subtoken_map"]

                    # Update the number of clusters
                    num_gt_clusters += len(gold_clusters)
                    num_pred_clusters += len(predicted_clusters)

                    oracle_clusters = action_sequences_to_clusters(gt_actions, pred_mentions)
                    oracle_clusters, mention_to_oracle = \
                        get_mention_to_cluster(oracle_clusters, threshold=cluster_threshold)
                    evaluator.update(predicted_clusters, gold_clusters,
                                     mention_to_predicted, mention_to_gold)
                    oracle_evaluator.update(oracle_clusters, gold_clusters,
                                            mention_to_oracle, mention_to_gold)

                    log_example["gt_actions"] = gt_actions
                    log_example["pred_actions"] = action_list
                    log_example["predicted_clusters"] = predicted_clusters

                    f.write(json.dumps(log_example) + "\n")

                print("Ground Truth Actions:", gt_class_counter)
                print("Predicted Actions:", pred_class_counter)

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

                logger.info("Action accuracy: %.3f, Oracle F-score: %.3f" %
                            (corr_actions/(total_actions + 1e-8), oracle_evaluator.get_prf()[2]))
                logger.info(log_file)
                logger.handlers[0].flush()

        logging.info("Inference time: %.2f" % inference_time)

        return result_dict

    def final_eval(self):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        perf_file = path.join(self.model_dir, "perf.json")
        if self.slurm_id:
            parent_dir = path.dirname(path.normpath(self.model_dir))
            perf_dir = path.join(parent_dir, "perf")
            if not path.exists(perf_dir):
                os.makedirs(perf_dir)
            perf_file = path.join(perf_dir, self.slurm_id + ".json")

        output_dict = {'model_dir': self.model_dir}
        for key, val in vars(self.args).items():
            output_dict[key] = val

        for split in ['dev', 'test']:
            # if self.train_with_singletons:
            #     cluster_thresholds = [1, 2]
            # else:
            #     cluster_thresholds = [2]
            cluster_thresholds = [self.canonical_cluster_threshold]
            # if self.cluster_threshold != self.canonical_cluster_threshold:
            # cluster_thresholds = [1, 2]
            for cluster_threshold in cluster_thresholds:
                logging.info('\n')
                logging.info('%s' % split.capitalize())
                result_dict = self.eval_model(split, final_eval=True, cluster_threshold=cluster_threshold)
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
            'model_args': self.model_args,
        }
        if model_type != 'best':
            param_groups = ['mem', 'doc'] if self.finetune else ['mem']
            for param_group in param_groups:
                save_dict['optimizer'][param_group] = self.optimizer[param_group].state_dict()
                save_dict['scheduler'][param_group] = self.optim_scheduler[param_group].state_dict()

        torch.save(save_dict, location)
        logging.info(f"Model saved at: {location}")
