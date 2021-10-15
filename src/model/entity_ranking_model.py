import torch
import torch.nn as nn

from model.mention_proposal import MentionProposalModule
from pytorch_utils.label_smoothing import LabelSmoothingLoss
from model.utils import get_gt_actions
from model.memory.entity_memory import EntityMemory


from typing import Dict, List, Tuple
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedTokenizerFast


class EntityRankingModel(nn.Module):
	"""
	Coreference model based on Entity-Ranking paradigm.

	In the entity-ranking paradigm, given a new mention we rank the different
	entity clusters to determine the clustering updates. Entity-Ranking paradigm
	allows for a naturally scalable solution to coreference resolution.
	Reference: Rahman and Ng [https://arxiv.org/pdf/1405.5202.pdf]

	This particular implementation represents the entities/clusters via fixed-dimensional
	dense representations, typically a simple avereage of mention representations.
	Clustering is performed in an online, autoregressive manner where mentions are
	processed in a left-to-right manner.
	References:
		Toshniwal et al [https://arxiv.org/pdf/2010.02807.pdf]
	  Toshniwal et al [https://arxiv.org/pdf/2109.09667.pdf]
	"""

	def __init__(self, model_config: DictConfig, train_config: DictConfig):
		super(EntityRankingModel, self).__init__()
		self.config = model_config
		self.train_config = train_config

		# Dropout module - Used during training
		self.drop_module = nn.Dropout(p=train_config.dropout_rate)

		# Document encoder + Mention proposer
		self.mention_proposer = MentionProposalModule(
			self.config, train_config, drop_module=self.drop_module)

		# Clustering module
		span_emb_size: int = self.mention_proposer.span_emb_size
		# Use of genre feature in clustering or not
		if self.config.metadata_params.use_genre_feature:
			self.config.memory.num_feats = 3

		self.memory_net = EntityMemory(
			config=self.config.memory, span_emb_size=span_emb_size, drop_module=self.drop_module)

	@property
	def device(self) -> torch.device:
		return self.mention_proposer.device

	def get_params(self, named=False) -> Tuple[List, List]:
		"""Returns a tuple of document encoder parameters and rest of the model params."""

		encoder_params, mem_params = [], []
		for name, param in self.named_parameters():
			elem = (name, param) if named else param
			if 'doc_encoder' in name:
				encoder_params.append(elem)
			else:
				mem_params.append(elem)

		return encoder_params, mem_params

	def get_tokenizer(self) -> PreTrainedTokenizerFast:
		"""Returns tokenizer used by the document encoder."""

		return self.mention_proposer.doc_encoder.get_tokenizer()

	def get_metadata(self, document: Dict) -> Dict:
		"""Extract metadata such as document genre from document."""

		meta_params = self.config.metadata_params
		if meta_params.use_genre_feature:
			doc_class = document["doc_key"][:2]
			if doc_class in meta_params.genres:
				doc_class_idx = meta_params.index(doc_class)
			else:
				doc_class_idx = meta_params.index(meta_params.default_genre)  # Default genre

			return {'genre': self.genre_embeddings(torch.tensor(doc_class_idx, device=self.device))}
		else:
			return {}

	def new_ignore_tuple_to_idx(self, action_tuple_list: List[Tuple[int, str]]) -> List:
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
			# action_indices = torch.tensor(action_indices, device=self.device)
			return action_indices
		else:
			return []

	def calculate_coref_loss(self, action_prob_list: List, action_tuple_list: List[Tuple[int, str]]):
		"""Calculates the coreference loss for the autoregressive online clustering module.

		Args:
			action_prob_list (List):
				Probability of each clustering action i.e. mention is merged with existing clusters
				or a new cluster is created.
			action_tuple_list (List[Tuple[int, str]]):
				Ground truth actions represented as a tuple of cluster index and action string.
				'c' represents that the mention is coreferent with existing clusters while
				'o' represents that the mention represents a new cluster.

		Returns:
			coref_loss (torch.Tensor):
				The scalar tensor representing the coreference loss.
		"""

		num_ents, counter = 0, 0
		coref_loss = 0.0

		if self.training:
			max_ents = self.config.memory.mem_type.max_ents
		else:
			max_ents = self.config.memory.mem_type.eval_max_ents

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
			# TODO(shtoshni): Use cross-entropy if label smoothing is not positive.
			label_smoothing_fn = LabelSmoothingLoss(smoothing=self.train_config.label_smoothing_wt, dim=0)

			coref_loss += label_smoothing_fn(pred=action_prob_list[counter], target=target, weight=weight)
			counter += 1

		return coref_loss

	def forward_training(self, document: Dict) -> Dict:
		"""Forward pass for training.

		Args:
			document: The tensorized document.

		Returns:
			loss_dict (Dict): Loss dictionary containing the losses of different stages of the model.
		"""

		# Initialize loss dictionary
		loss_dict = {'total': None}

		# Encode documents and get mentions
		proposer_output_dict = self.mention_proposer(document)
		pred_mentions = proposer_output_dict.get('ments', None)
		# Only cluster if there are mentions to cluster over
		if pred_mentions is None:
			return loss_dict

		mention_emb_list = proposer_output_dict['ment_emb_list']
		pred_mentions_list = pred_mentions.tolist()

		# Get ground truth clustering mentions
		gt_actions: List[Tuple[int, str]] = get_gt_actions(pred_mentions_list, document)

		# Metadata such as document genre can be used by model for clustering
		metadata: Dict = self.get_metadata(document)

		# Perform teacher-forced clustering to get gt action probabilities
		coref_new_list: List[Tensor] = self.memory_net.forward_training(
			pred_mentions, mention_emb_list, gt_actions, metadata)

		# Consolidate different losses in one dictionary
		if 'ment_loss' in proposer_output_dict:
			loss_dict = {'total': proposer_output_dict['ment_loss'], 'entity': proposer_output_dict['ment_loss']}
		else:
			loss_dict = {'total': 0.0}

		if len(coref_new_list) > 0:
			coref_loss = self.calculate_coref_loss(coref_new_list, gt_actions)
			loss_dict['total'] += coref_loss
			loss_dict['coref'] = coref_loss

		return loss_dict

	def forward(self, document: Dict) -> Tuple[List, List, List, List]:
		"""Forward pass of the streaming coreference model.

		This method performs streaming coreference. The entity clusters from previous
		documents chunks are represented as vectors and passed along to the processing
		of subsequent chunks along with the metadata associated with these clusters.

		Args:
			document (Dict): Tensorized document

		Returns:
			 pred_mentions_list (List): Mentions predicted by the mention proposal module
			 mention_scores (List): Scores assigned by the mention proposal module for
			      the predicted mentions
			 gt_actions (List): Ground truth clustering actions; useful for calculating oracle performance
			 action_list (List): Actions predicted by the clustering module for the predicted mentions
		'"""

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
		return pred_mentions_list,  mention_scores, gt_actions, action_list
