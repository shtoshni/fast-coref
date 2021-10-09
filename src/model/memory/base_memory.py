import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
import math

LOG2 = math.log(2)


class BaseMemory(nn.Module):
	def __init__(self, config, span_emb_size, drop_module):
		super(BaseMemory, self).__init__()
		self.config = config
		# TODO - num feats

		self.mem_size = span_emb_size
		self.drop_module = drop_module

		if self.config.sim_func == 'endpoint':
			num_embs = 2  # Span start, Span end
		else:
			num_embs = 3  # Span start, Span end, Hadamard product between the two

		self.mem_coref_mlp = MLP(
			num_embs * self.mem_size + config.num_feats * config.emb_size,
			config.mlp_size, 1, drop_module=drop_module,
			num_hidden_layers=config.mlp_depth, bias=True)

		if config.entity_rep == 'learned_avg':
			# Parameter for updating the cluster representation
			self.alpha = MLP(
				2 * self.mem_size, config.mlp_size, 1, num_hidden_layers=1, bias=True, drop_module=drop_module)

		self.distance_embeddings = nn.Embedding(10, config.emb_size)
		self.counter_embeddings = nn.Embedding(10, config.emb_size)

	def initialize_memory(self, mem=None, ent_counter=None, last_mention_start=None):
		if mem is None:
			mem = torch.zeros(1, self.mem_size).to(self.device)
			ent_counter = torch.tensor([0.0]).to(self.device)
			last_mention_start = torch.zeros(1).long().to(self.device)

		return mem, ent_counter, last_mention_start

	@staticmethod
	def get_bucket(count):
		"Bucket distance and entity counters using the same logic."
		logspace_idx = torch.floor(torch.log(count.float()) / LOG2).long() + 3
		use_identity = (count <= 4).long()
		combined_idx = use_identity * count + (1 - use_identity) * logspace_idx
		return torch.clamp(combined_idx, 0, 9)

	@staticmethod
	def get_distance_bucket(distances):
		return BaseMemory.get_bucket(distances)

	@staticmethod
	def get_counter_bucket(count):
		return BaseMemory.get_bucket(count)

	def get_distance_emb(self, distance):
		distance_tens = self.get_distance_bucket(distance)
		distance_embs = self.distance_embeddings(distance_tens)
		return distance_embs

	def get_counter_emb(self, ent_counter):
		counter_buckets = self.get_counter_bucket(ent_counter.long())
		counter_embs = self.counter_embeddings(counter_buckets)
		return counter_embs

	@staticmethod
	def get_coref_mask(ent_counter):
		cell_mask = (ent_counter > 0.0).float()
		return cell_mask

	def get_feature_embs(self, ment_start, last_mention_start, ent_counter, metadata):
		distance_embs = self.get_distance_emb(ment_start - last_mention_start)
		counter_embs = self.get_counter_emb(ent_counter)

		feature_embs_list = [distance_embs, counter_embs]

		if 'genre' in metadata:
			genre_emb = metadata['genre']
			num_ents = distance_embs.shape[0]
			genre_emb = torch.unsqueeze(genre_emb, dim=0).repeat(num_ents, 1)
			feature_embs_list.append(genre_emb)

		feature_embs = self.drop_module(torch.cat(feature_embs_list, dim=-1))
		return feature_embs

	def get_coref_new_scores(self, ment_emb, mem_vectors, ent_counter, feature_embs, ment_score=0):
		# Repeat the query vector for comparison against all cells
		num_ents = mem_vectors.shape[0]
		rep_ment_emb = ment_emb.repeat(num_ents, 1)  # M x H

		# Coref Score
		if self.config.sim_func == 'endpoint':
			pair_vec = torch.cat([mem_vectors, rep_ment_emb, feature_embs], dim=-1)
			pair_score = self.mem_coref_mlp(pair_vec)
		else:
			pair_vec = torch.cat(
				[mem_vectors, rep_ment_emb, mem_vectors * rep_ment_emb, feature_embs], dim=-1)
			pair_score = self.mem_coref_mlp(pair_vec)

		coref_score = torch.squeeze(pair_score, dim=-1) + ment_score  # M

		coref_new_mask = torch.cat([self.get_coref_mask(ent_counter), torch.tensor([1.0], device=self.device)], dim=0)
		coref_new_score = torch.cat(([coref_score, torch.tensor([0.0], device=self.device)]), dim=0)
		coref_new_score = coref_new_score * coref_new_mask + (1 - coref_new_mask) * (-1e4)
		return coref_new_score

	@staticmethod
	def assign_cluster(coref_new_scores):
		num_ents = coref_new_scores.shape[0] - 1
		pred_max_idx = torch.argmax(coref_new_scores).item()
		if pred_max_idx < num_ents:
			# Coref
			return pred_max_idx, 'c'
		else:
			# New cluster
			return num_ents, 'o'

	def coref_update(self, ment_emb, mem_vectors, cell_idx, ent_counter):
		if self.config.entity_rep == 'learned_avg':
			alpha_wt = torch.sigmoid(
				self.alpha(torch.cat([mem_vectors[cell_idx, :], ment_emb], dim=0)))
			coref_vec = alpha_wt * mem_vectors[cell_idx, :] + (1 - alpha_wt) * ment_emb
		elif self.config.entity_rep == 'max':
			coref_vec = torch.max(mem_vectors[cell_idx], ment_emb)
		else:
			cluster_count = ent_counter[cell_idx].item()
			coref_vec = (mem_vectors[cell_idx] * cluster_count + ment_emb) / (cluster_count + 1)

		return coref_vec
