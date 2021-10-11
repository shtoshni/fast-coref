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

from data_utils.utils import load_dataset, load_eval_dataset
import pytorch_utils.utils as utils

from model.entity_ranking_model import EntityRankingModel
from data_utils.tensorize_dataset import TensorizeDataset
from pytorch_utils.optimization_utils import get_inverse_square_root_decay

from utils_evaluate import coref_evaluation

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


class Experiment:
	def __init__(self, config):
		self.config = config

		# Whether to train or not
		self.eval_model = (not self.config.train)

		# Step 1 - Build model
		self._build_model()

		# Step 2 - Load Data - Data processing choices such as tokenizer will depend on the model
		self._load_data()

		# Step 3 - Load model and resume training if required

		# Initialize dictionary to track key training variables
		self.train_info = {'val_perf': 0.0, 'global_steps': 0, 'num_stuck_evals': 0, 'peak_memory': 0.0}

		if self.eval_model:
			# Load the best checkpoint
			self._load_previous_checkpoint(last_checkpoint=False)
		else:
			# Resume training
			self._setup_training()
			# Loading the checkpoint also restores the training metadata
			self._load_previous_checkpoint(last_checkpoint=True)

			# All set to resume training
			# But first check if training is remaining
			if self._is_training_remaining():
				self.train()

		# Step 4 - Perform final evaluation
		self.load_model(self.best_model_path, last_checkpoint=False)
		self.perform_final_eval()

	def _build_model(self):
		model_params = self.config.model
		train_config = self.config.trainer
		self.model = EntityRankingModel(config=model_params,  train_config=train_config)
		if torch.cuda.is_available():
			self.model.cuda()

	def _load_data(self):
		self.orig_data_map, self.num_train_docs_map, self.data_iter_map = {}, {}, {}
		self.conll_data_dir = {}

		eval_model = (not self.config.train)
		max_segment_len = self.config.model.doc_encoder.transformer.max_segment_len
		model_name: str = self.config.model.doc_encoder.transformer.name
		add_speaker_tokens = self.config.model.doc_encoder.add_speaker_tokens
		base_data_dir = path.abspath(self.config.paths.base_data_dir)

		# Load data
		for dataset_name, attributes in self.config.datasets.items():
			num_train_docs = attributes.get('num_train_docs', None)
			num_eval_docs = attributes.get('num_eval_docs', None)
			num_test_docs = attributes.get('num_test_docs', None)
			singleton_file = attributes.get('singleton_file', None)

			# Data directory is a function of dataset name and tokenizer used
			data_dir = path.join(path.join(base_data_dir, dataset_name), model_name)
			# Check if speaker tokens are added
			if add_speaker_tokens:
				pot_data_dir = path.join(
					path.join(path.join(base_data_dir, dataset_name)), model_name + '_speaker')
				if path.exists(pot_data_dir):
					data_dir = pot_data_dir

			# Datasets such as litbank have cross validation splits
			if attributes.get('cross_val_split', None) is not None:
				data_dir = path.join(data_dir, str(attributes.get('cross_val_split')))

			# CoNLL data dir
			if attributes.get('has_conll', False):
				conll_dir = path.join(path.join(path.join(base_data_dir, dataset_name)), 'conll')
				if attributes.get('cross_val_split', None) is not None:
					# LitBank like datasets have cross validation splits
					conll_dir = path.join(conll_dir, str(attributes.get('cross_val_split')))

				if path.exists(conll_dir):
					self.conll_data_dir[dataset_name] = conll_dir

			if eval_model:
				self.orig_data_map[dataset_name] = load_eval_dataset(
					data_dir, max_segment_len=max_segment_len,
					num_test_docs=num_test_docs
				)
			else:
				self.num_train_docs_map[dataset_name] = attributes.get('num_train_docs', None)
				self.orig_data_map[dataset_name] = load_dataset(
					data_dir, singleton_file=singleton_file,
					num_train_docs=num_train_docs, num_eval_docs=num_eval_docs,
					num_test_docs=num_test_docs,
				)

		# Tensorize data
		data_processor = TensorizeDataset(
			self.model.get_tokenizer(), remove_singletons=(not self.config.keep_singletons))

		if eval_model:
			self.data_iter_map['test'] = {}
			for dataset in self.orig_data_map:
				self.data_iter_map['test'][dataset] = \
					data_processor.tensorize_data(self.orig_data_map[dataset]['test'])
		else:
			# Training
			for split in ['train', 'dev', 'test']:
				self.data_iter_map[split] = {}
				training = (split == 'train')
				for dataset in self.orig_data_map:
					self.data_iter_map[split][dataset] = \
						data_processor.tensorize_data(self.orig_data_map[dataset][split], training=training)

			# Estimate number of training steps
			if self.config.trainer.eval_per_k_steps is None:
				# Eval steps is 1 epoch (with subsampling) of all the datasets used in joint training
				self.config.trainer.eval_per_k_steps = sum(self.num_train_docs_map.values())

			self.config.trainer.num_training_steps = self.config.trainer.eval_per_k_steps * self.config.trainer.max_evals
			logger.info(f"Number of training steps: {self.config.trainer.num_training_steps}")

	def _load_previous_checkpoint(self, last_checkpoint=True):
		conf_paths = self.config.paths

		if (self.config.paths.model_path is None) or (not path.exists(self.config.paths.model_path)):
			# Model path is specified via CLI - Probably for evaluation
			self.model_path = self.config.paths.model_path
			self.best_model_path = self.config.paths.model_path
		else:
			self.model_path = path.join(conf_paths.model_dir, conf_paths.model_file)
			self.best_model_path = path.join(conf_paths.best_model_dir, conf_paths.model_file)

		if last_checkpoint:
			# Resume training
			if path.exists(self.model_path):
				self.load_model(self.model_path, last_checkpoint=last_checkpoint)
			else:
				# Starting training
				torch.random.manual_seed(self.config.seed)
				np.random.seed(self.config.seed)
				random.seed(self.config.seed)
		else:
			# Load best model
			if path.exists(self.best_model_path):
				self.load_model(self.best_model_path, last_checkpoint=last_checkpoint)
			else:
				raise IOError(f"Best model path at {self.best_model_path} not found")

			logger.info("\nModel loaded\n")
			utils.print_model_info(self.model)
			sys.stdout.flush()

	def _is_training_remaining(self):
		if self.train_info['num_stuck_evals'] >= self.config.trainer.patience:
			return False
		if self.train_info['global_steps'] >= self.config.trainer.num_training_steps:
			return False

		return True

	def _setup_training(self):
		# Dictionary to track key training variables
		self.train_info = {'val_perf': 0.0, 'global_steps': 0, 'num_stuck_evals': 0, 'peak_memory': 0.0}

		# Initialize optimizers
		self._initialize_optimizers()

	def _initialize_optimizers(self):
		"""Initialize model + optimizer(s). Check if there's a checkpoint in which case we resume from there."""
		optimizer_config = self.config.optimizer
		train_config = self.config.trainer
		self.optimizer, self.optim_scheduler = {}, {}

		if torch.cuda.is_available():
			# Gradient scaler required for mixed precision training
			self.scaler = torch.cuda.amp.GradScaler()
		else:
			self.scaler = None

		# Optimizer for clustering params
		self.optimizer['mem'] = torch.optim.Adam(
			self.model.get_params()[1], lr=optimizer_config.init_lr, eps=1e-6)

		if optimizer_config.lr_decay == 'inv':
			self.optim_scheduler['mem'] = get_inverse_square_root_decay(self.optimizer['mem'], num_warmup_steps=0)
		else:
			# No warmup steps for model params
			self.optim_scheduler['mem'] = get_linear_schedule_with_warmup(
				self.optimizer['mem'], num_warmup_steps=0, num_training_steps=train_config.num_training_steps)

		if optimizer_config.fine_tune_lr is not None:
			# Optimizer for document encoder
			no_decay = ['bias', 'LayerNorm.weight']  # No weight decay for bias and layernorm weights
			encoder_params = self.model.get_params(named=True)[0]
			grouped_param = [
				{'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
				 'lr': optimizer_config.fine_tune_lr, 'weight_decay': 1e-2},
				{'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
				 'lr': optimizer_config.fine_tune_lr,
				 'weight_decay': 0.0}
			]

			self.optimizer['doc'] = AdamW(grouped_param, lr=optimizer_config.fine_tune_lr, eps=1e-6)

			# Scheduler for document encoder
			num_warmup_steps = int(0.1 * train_config.num_training_steps)
			if optimizer_config.lr_decay == 'inv':
				self.optim_scheduler['doc'] = get_inverse_square_root_decay(
					self.optimizer['doc'], num_warmup_steps=num_warmup_steps)
			else:
				self.optim_scheduler['doc'] = get_linear_schedule_with_warmup(
					self.optimizer['doc'], num_warmup_steps=num_warmup_steps,
					num_training_steps=train_config.num_training_steps)

	def train(self):
		"""Train model"""
		model, optimizer, scheduler, scaler = self.model, self.optimizer, self.optim_scheduler, self.scaler
		model.train()

		optimizer_config, train_config = self.config.optimizer, self.config.trainer

		start_time = time.time()
		eval_time = {'total_time': 0, 'num_evals': 0}
		while True:
			logger.info("Steps done %d" % (self.train_info['global_steps']))

			train_data = []
			for dataset, dataset_train_data in self.data_iter_map['train'].items():
				np.random.shuffle(dataset_train_data)
				if self.num_train_docs_map.get(dataset, None) is not None:
					train_data += dataset_train_data[:self.num_train_docs_map[dataset]]
				else:
					train_data += dataset_train_data
			np.random.shuffle(train_data)
			logger.info("Per epoch training steps: %d" % len(train_data))
			encoder_params, task_params = model.get_params()

			for cur_example in train_data:
				def handle_example(example):
					self.train_info['global_steps'] += 1
					for key in optimizer:
						optimizer[key].zero_grad()

					if scaler is not None:
						with torch.cuda.amp.autocast():
							loss = model.forward_training(example)
							total_loss = loss['total']
							if total_loss is None:
								return None

							scaler.scale(total_loss).backward()
							for key in optimizer:
								scaler.unscale_(optimizer[key])
					else:
						loss = model.forward_training(example)
						total_loss = loss['total']
						if total_loss is None:
							return None

					# Gradient clipping
					torch.nn.utils.clip_grad_norm_(encoder_params, optimizer_config.max_gradient_norm)
					torch.nn.utils.clip_grad_norm_(task_params, optimizer_config.max_gradient_norm)

					for key in optimizer:
						# Optimizer step
						if scaler is not None:
							scaler.step(optimizer[key])
						else:
							optimizer[key].step()

						# Scheduler step
						scheduler[key].step()

					# Update scaler
					if scaler is not None:
						scaler.update()

					return total_loss.item()

				example_loss = handle_example(cur_example)

				if self.train_info['global_steps'] % train_config.log_frequency == 0:
					logger.info('{} {:.3f} Max mem {:.3f} GB'.format(
						cur_example["doc_key"], example_loss,
						(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0)
					)
					torch.cuda.reset_peak_memory_stats()

				if train_config.eval_per_k_steps and \
							(self.train_info['global_steps'] % train_config.eval_per_k_steps == 0):
					fscore = self.periodic_model_eval()
					# Get elapsed time
					elapsed_time = time.time() - start_time

					start_time = time.time()
					logger.info(
						"Steps: %d, F1: %.1f, Max F1: %.1f, Time: %.2f"
						% (self.train_info['global_steps'], fscore, self.train_info['val_perf'], elapsed_time))

					# Check stopping criteria
					if self.train_info['num_stuck_evals'] >= train_config.patience:
						return
					if self.train_info['global_steps'] >= train_config.num_training_steps:
						return

					if not self.config.infra.is_local:
						# Check if enough time on cluster to run another eval
						eval_time['total_time'] += elapsed_time
						eval_time['num_evals'] += 1

						avg_eval_time = eval_time['total_time'] / eval_time['num_evals']
						rem_time = self.config.infra.job_time - eval_time['total_time']
						logging.info("Average eval time: %.2f mins, Remaining time: %.2f mins"
						             % (avg_eval_time / 60, rem_time / 60))

						if rem_time < avg_eval_time:
							logging.info('Canceling job as not much time left')
							sys.exit()

			logger.handlers[0].flush()

	def periodic_model_eval(self):
		# Dev performance
		fscore_dict = {}
		for dataset in self.data_iter_map['dev']:
			result_dict = coref_evaluation(self.config, self.model, self.data_iter_map, dataset)
			fscore_dict[dataset] = result_dict.get('fscore', 0.0)

		logger.info(fscore_dict)
		# Calculate Mean F-score
		fscore = sum([fscore_dict[dataset] for dataset in fscore_dict]) / len(fscore_dict)
		logger.info('F1: %.1f, Max F1: %.1f' % (fscore, self.train_info['val_perf']))
		# Update model if dev performance improves
		if fscore > self.train_info['val_perf']:
			self.train_info['num_stuck_evals'] = 0
			self.train_info['val_perf'] = fscore
			logger.info('Saving best model')
			self.save_model(self.best_model_path, model_type='best')
		else:
			self.train_info['num_stuck_evals'] += 1

		# Save model
		if self.config.trainer.to_save_model:
			self.save_model(self.model_path)

		# Go back to training mode
		self.model.train()
		return fscore

	def perform_final_eval(self):
		"""Evaluate the model on train, dev, and test"""
		base_output_dict = dict(self.config)
		perf_summary = {
			'model_dir': path.normpath(self.config.paths.model_dir), 'best_perf': self.train_info['val_perf']}
		logging.info("Validation performance: %.1f" % self.train_info['val_perf'])

		for split in ['test']:
			logger.info('\n')
			logger.info('%s' % split.capitalize())

			for dataset in self.data_iter_map[split]:
				dataset_dir = path.join(self.config.paths.model_dir, dataset)
				if not path.exists(dataset_dir):
					os.makedirs(dataset_dir)
				perf_file = path.join(dataset_dir, "perf.json")

				logger.info('\n')
				logger.info('%s\n' % self.config.datasets[dataset].name)

				result_dict = coref_evaluation(
					self.config, self.model, self.data_iter_map, dataset=dataset, split='test', final_eval=True,
					conll_data_dir=self.conll_data_dir
				)

				output_dict = dict(base_output_dict)
				output_dict[f"{dataset}_{split}"] = result_dict
				perf_summary[f"{dataset}_{split}"] = result_dict['fscore']

				json.dump(output_dict, open(perf_file, 'w'), indent=2)

				logging.info("Final performance summary at %s" % perf_file)
				sys.stdout.flush()

		summary_file = path.join(self.config.paths.model_dir, 'perf.json')

		# Change paths if running on cluster (slurm)
		if not self.config.infra.is_local:
			parent_dir = path.dirname(path.normpath(self.config.paths.model_dir))
			perf_dir = path.join(parent_dir, "perf")
			if not path.exists(perf_dir):
				os.makedirs(perf_dir)

			gold_ment_str = ''
			if self.model.use_gold_ments:
				gold_ment_str = '_gold'
			summary_file = path.join(perf_dir, self.config.infra.job_id + gold_ment_str + ".json")

		json.dump(perf_summary, open(summary_file, 'w'), indent=2)
		logger.info("Performance summary file: %s" % summary_file)

	def load_model(self, location, last_checkpoint=True):
		checkpoint = torch.load(location, map_location='cpu')
		self.model.load_state_dict(checkpoint['model'], strict=False)
		if last_checkpoint:
			for param_group in checkpoint['optimizer']:
				self.optimizer[param_group].load_state_dict(
					checkpoint['optimizer'][param_group])
				self.optim_scheduler[param_group].load_state_dict(
					checkpoint['scheduler'][param_group])

			if 'scaler' in checkpoint and self.scaler is not None:
				self.scaler.load_state_dict(checkpoint['scaler'])

		if last_checkpoint:
			self.train_info = checkpoint['train_info']
			torch.set_rng_state(checkpoint['rng_state'])
			np.random.set_state(checkpoint['np_rng_state'])

	def save_model(self, location, model_type='last'):
		"""Save model"""
		model_state_dict = OrderedDict(self.model.state_dict())
		if not self.config.model.doc_encoder.finetune:
			for key in self.model.state_dict():
				if 'lm_encoder.' in key:
					del model_state_dict[key]

		save_dict = {
			'train_info': self.train_info,
			'model': model_state_dict,
			'rng_state': torch.get_rng_state(),
			'np_rng_state': np.random.get_state(),
			'optimizer': {}, 'scheduler': {},
			'config': self.config,
		}

		if self.scaler is not None:
			save_dict['scaler'] = self.scaler.state_dict()

		if model_type != 'best':
			param_groups = ['mem', 'doc'] if self.config.model.doc_encoder.finetune else ['mem']
			for param_group in param_groups:
				save_dict['optimizer'][param_group] = self.optimizer[param_group].state_dict()
				save_dict['scheduler'][param_group] = self.optim_scheduler[param_group].state_dict()

		torch.save(save_dict, location)
		logging.info(f"Model saved at: {location}")
