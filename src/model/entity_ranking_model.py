import torch
import torch.nn as nn

from model.mention_proposal import MentionProposalModule
from pytorch_utils.label_smoothing import LabelSmoothingLoss
from model.utils import get_gt_actions
from model.memory.entity_memory import EntityMemory


class EntityRankingModel(nn.Module):
	def __init__(self, model_config, dropout_rate=0.0):
		super(EntityRankingModel, self).__init__()
		self.config = model_config

		# Dropout module - Used during training
		self.drop_module = nn.Dropout(p=dropout_rate)

		# Document encoder + Mention proposer
		self.mention_proposer = MentionProposalModule(model_config, drop_module=self.drop_module)

		# Clustering module
		span_emb_size = self.mention_proposer.span_emb_size
		# Use of genre feature in clustering or not
		if self.config.metadata_params.use_genre_feature:
			model_config.memory.num_feats = 3
		self.memory = EntityMemory(
			config=model_config.memory, span_emb_size=span_emb_size, drop_module=self.drop_module)

	def get_tokenizer(self):
		return self.mention_proposer.doc_encoder.get_tokenizer()

	def get_metadata(self, document):
		meta_params = self.config.metada_params
		if meta_params.use_genre_feature:
			doc_class = document["doc_key"][:2]
			if doc_class in meta_params.genres:
				doc_class_idx = meta_params.index(doc_class)
			else:
				doc_class_idx = meta_params.index(meta_params.default_genre)  # Default genre

			return {'genre': self.genre_embeddings(torch.tensor(doc_class_idx, device=self.device))}
		else:
			return {}

	def new_ignore_tuple_to_idx(self, action_tuple_list):
		action_indices = []
		max_ents = (self.config.max_ents if self.training else self.config.eval_max_ents)

		for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
			if action_str == 'o':
				if self.mem_type == 'lru':
					action_indices.append(0)
				else:
					action_indices.append(cell_idx)
			elif action_str == 'n':
				# No space
				if self.mem_type == 'lru':
					action_indices.append(1)
				else:
					action_indices.append(max_ents)

		# The first max_ents are all overwrites - We skip that part
		if len(action_indices) > max_ents:
			action_indices = action_indices[max_ents:]
			action_indices = torch.tensor(action_indices, device=self.device)
			return action_indices
		else:
			return []

	def calculate_coref_loss(self, action_prob_list, action_tuple_list):
		"Calculates the coreference loss given the action probability list and ground truth actions."
		num_ents, counter = 0, 0
		coref_loss = 0.0
		max_ents = (self.config.max_ents if self.training else self.config.eval_max_ents)

		for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
			if action_str == 'c':
				# Coreference with clusters currently tracked
				gt_idx = cell_idx

			elif action_str == 'o':
				# Overwrite - New cluster
				gt_idx = num_ents
				if max_ents is None or num_ents < max_ents:
					num_ents += 1

				if num_ents == 1:  # The first ent is always overwritten - No loss there
					continue
			else:
				continue

			target = torch.tensor([gt_idx], device=self.device)
			weight = torch.ones_like(action_prob_list[counter], device=self.device)
			label_smoothing_fn = LabelSmoothingLoss(smoothing=self.label_smoothing_wt, dim=0)

			coref_loss += label_smoothing_fn(pred=action_prob_list[counter], target=target, weight=weight)
			counter += 1

		return coref_loss

	def forward_training(self, document):
		# Initialize loss dictionary
		loss_dict = {'total': None}

		# Encode documents and get mentions
		proposer_output_dict = self.mention_proposer(document)

		pred_mentions = proposer_output_dict.get('ments', None)
		if pred_mentions is None:
			return loss_dict

		mention_emb_list = proposer_output_dict['ment_emb_list']

		pred_mentions_list = pred_mentions.tolist()
		gt_actions = get_gt_actions(pred_mentions_list, document)

		metadata = self.get_metadata(document)
		coref_new_list = self.memory_net.forward_training(pred_mentions, mention_emb_list, gt_actions, metadata)

		if 'ment_loss' in proposer_output_dict:
			loss_dict = {'total': proposer_output_dict['ment_loss'], 'entity': proposer_output_dict['ment_loss']}

		if len(coref_new_list) > 0:
			coref_loss = self.calculate_coref_loss(coref_new_list, gt_actions)
			loss_dict['total'] += coref_loss
			loss_dict['coref'] = coref_loss

		return loss_dict

	def forward(self, document):
		'''
		Process document chunk by chunk.
		Pass along the previous clusters
		'''
		# Initialize lists to track all the actions taken, mentions predicted across the chunks
		action_list, pred_mentions_list, gt_actions, mention_scores = [], [], [], []
		# Initialize entity clusters and current document token offset
		entity_cluster_states, token_offset = None, 0

		metadata = self.get_metadata(document)

		for idx in range(0, len(document["sentences"])):
			num_tokens = len(document["sentences"][idx])

			clusters = []
			for orig_cluster in document.get("clusters", []):
				cluster = []
				for mention in orig_cluster:
					ment_start, ment_end = mention[:2]
					if ment_start >= token_offset and ment_end < (token_offset + num_tokens):
						cluster.append((ment_start - token_offset, ment_end - token_offset))

				if len(cluster):
					clusters.append(cluster)

			cur_example = {
				"tensorized_sent": document["tensorized_sent"][idx],
				"sentence_map": document["sentence_map"][token_offset: token_offset + num_tokens],
				"subtoken_map": document["subtoken_map"][token_offset: token_offset + num_tokens],
				"sent_len_list": [document["sent_len_list"][idx]],
				"clusters": clusters,
			}

			# Pass along other metadata
			for key in document:
				if key not in cur_example:
					cur_example[key] = document[key]

			proposer_output_dict = self.mention_proposer(cur_example)
			if proposer_output_dict.get('ments', None) is None:
				token_offset += num_tokens
				continue
			cur_pred_mentions = proposer_output_dict.get('ments') + token_offset
			token_offset += num_tokens

			pred_mentions_list.extend(cur_pred_mentions.tolist())
			mention_scores.extend(proposer_output_dict['ment_scores'].tolist())

			# Pass along entity clusters from previous chunks while processing next chunks
			cur_action_list, entity_cluster_states = self.memory_net(
				cur_pred_mentions, proposer_output_dict['ment_emb_list'], metadata,
				memory_init=entity_cluster_states)
			action_list.extend(cur_action_list)

		gt_actions = get_gt_actions(pred_mentions_list, document)
		return action_list, pred_mentions_list, gt_actions, mention_scores
