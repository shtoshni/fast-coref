import torch
from model.memory import BaseMemory
from pytorch_utils.modules import MLP
import torch.nn as nn

from omegaconf import DictConfig
from typing import Dict, Tuple, List
from torch import Tensor as T


class EntityMemory(BaseMemory):
	def __init__(self, config: DictConfig, span_emb_size: int, drop_module: nn.Module) -> None:
		super(EntityMemory, self).__init__(config, span_emb_size, drop_module)
		self.mem_type: DictConfig = config.mem_type

		# Check if the memory is bounded i.e. there's a limit on the number of max entities
		self.is_mem_bounded: bool = config.mem_type.max_ents is not None
		if self.is_mem_bounded:
			self.fert_mlp = MLP(
				input_size=self.mem_size + config.num_feats * self.emb_size, bias=True,
				hidden_size=config.mlp_size, output_size=1, num_hidden_layers=config.mlp_depth,
				drop_module=drop_module)

	def predict_new_or_ignore_learned(
					self, ment_emb: T, mem_vectors: T,
					feature_embs: T, ment_feature_embs: T) -> Tuple[T, int, str]:
		# Fertility Score
		mem_fert_input = torch.cat([mem_vectors, feature_embs], dim=-1)
		ment_fert_input = torch.unsqueeze(torch.cat([ment_emb, ment_feature_embs], dim=-1), dim=0)
		fert_input = torch.cat([mem_fert_input, ment_fert_input], dim=0)

		fert_scores: T = self.fert_mlp(fert_input)
		fert_scores = torch.squeeze(fert_scores, dim=-1)

		min_idx = int(torch.argmin(fert_scores).item())
		max_ents = (self.config.max_ents if self.training else self.config.eval_max_ents)
		if min_idx < max_ents:
			# The fertility of one of the entities currently being tracked is lower than the new entity
			# We will overwrite this entity
			output = (fert_scores, min_idx, 'o',)
		else:
			# No space - The new entity is not "fertile" enough
			output = (fert_scores, -1, 'n',)

		return output

	def predict_new_or_ignore_lru(
					self, ment_emb: T, mem_vectors: T, feature_embs: T,
					ment_feature_embs: T, lru_list: List[int]) -> Tuple[T, int, str]:
		lru_cell = lru_list[0]
		mem_fert_input = torch.cat([mem_vectors[lru_cell, :], feature_embs[lru_cell, :]], dim=0)
		ment_fert_input = torch.cat([ment_emb, ment_feature_embs], dim=-1)
		fert_input = torch.stack([mem_fert_input, ment_fert_input], dim=0)
		fert_scores = torch.squeeze(self.fert_mlp(fert_input), dim=-1)
		output = fert_scores,

		over_max_idx = torch.argmax(fert_scores).item()
		if over_max_idx == 0:
			return output + (lru_cell, 'o',)
		else:
			# No space - The new entity is not "fertile" enough
			return output + (-1, 'n',)

	def forward_training(self, ment_boundaries: List[List[T]], mention_emb_list: List[T],
	                     gt_actions: List[Tuple[int, str]], metadata: Dict) -> List[T]:
		# Initialize memory
		first_overwrite, coref_new_list = True, []
		mem_vectors, ent_counter, last_mention_start = self.initialize_memory()

		for ment_idx, (ment_emb, (gt_cell_idx, gt_action_str)) in enumerate(zip(mention_emb_list, gt_actions)):
			ment_start, ment_end = ment_boundaries[ment_idx]

			if first_overwrite:
				first_overwrite = False
				mem_vectors = torch.unsqueeze(ment_emb, dim=0)
				ent_counter = torch.tensor([1.0], device=self.device)
				last_mention_start = torch.tensor([ment_start], dtype=torch.long, device=self.device)
				continue
			else:
				feature_embs = self.get_feature_embs(
					ment_start, last_mention_start, ent_counter, metadata)
				coref_new_scores = self.get_coref_new_scores(
					ment_emb, mem_vectors, ent_counter, feature_embs)
				coref_new_list.append(coref_new_scores)

			# Teacher forcing
			action_str, cell_idx = gt_action_str, gt_cell_idx

			num_ents: int = int(torch.sum((ent_counter > 0).long()).item())
			cell_mask: torch.Tensor = (
							torch.arange(start=0, end=num_ents, device=self.device) == torch.tensor(cell_idx)).float()
			mask = torch.unsqueeze(cell_mask, dim=1)
			mask = mask.repeat(1, self.mem_size)

			if action_str == 'c':
				coref_vec = self.coref_update(ment_emb, mem_vectors, cell_idx, ent_counter)
				mem_vectors = mem_vectors * (1 - mask) + mask * coref_vec
				ent_counter[cell_idx] = ent_counter[cell_idx] + 1
				last_mention_start[cell_idx] = ment_start
			elif action_str == 'o':
				# Append the new entity
				mem_vectors = torch.cat([mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0)
				ent_counter = torch.cat([ent_counter, torch.tensor([1.0], device=self.device)], dim=0)
				last_mention_start = torch.cat([last_mention_start, ment_start.unsqueeze(dim=0)], dim=0)

		return coref_new_list

	def forward(
					self, ment_boundaries: List[List[T]], mention_emb_list: torch.Tensor, metadata: Dict,
					memory_init: Dict = None) -> Tuple[List[Tuple[int, str]], Dict]:
		# Initialize memory
		if memory_init is not None:
			mem_vectors, ent_counter, last_mention_start = self.initialize_memory(**memory_init)
		else:
			mem_vectors, ent_counter, last_mention_start = self.initialize_memory()

		action_list = []  # argmax actions
		first_overwrite = (True if torch.sum(ent_counter) == 0 else False)

		for ment_idx, ment_emb in enumerate(mention_emb_list):
			ment_start, ment_end = ment_boundaries[ment_idx]
			feature_embs = self.get_feature_embs(ment_start, last_mention_start, ent_counter, metadata)

			if first_overwrite:
				pred_cell_idx, pred_action_str = 0, 'o'
			else:
				coref_new_scores = self.get_coref_new_scores(
					ment_emb, mem_vectors, ent_counter, feature_embs)

				pred_cell_idx, pred_action_str = self.assign_cluster(coref_new_scores)

			action_list.append((pred_cell_idx, pred_action_str))
			action_str, cell_idx = pred_action_str, pred_cell_idx

			if first_overwrite:
				first_overwrite = False
				# We start with a single empty memory cell
				mem_vectors = torch.unsqueeze(ment_emb, dim=0)
				ent_counter = torch.tensor([1.0], device=self.device)
				last_mention_start[0] = ment_start
			else:
				if action_str == 'c':
					coref_vec = self.coref_update(ment_emb, mem_vectors, cell_idx, ent_counter)
					mem_vectors[cell_idx] = coref_vec
					ent_counter[cell_idx] = ent_counter[cell_idx] + 1
					last_mention_start[cell_idx] = ment_start

				elif action_str == 'o':
					# Append the new entity
					mem_vectors = torch.cat([mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0)
					ent_counter = torch.cat([ent_counter, torch.tensor([1.0], device=self.device)], dim=0)
					last_mention_start = torch.cat(
						[last_mention_start, ment_start.unsqueeze(dim=0)], dim=0)

		mem_state = {
			"mem": mem_vectors, "ent_counter": ent_counter, "last_mention_start": last_mention_start}
		return action_list, mem_state
