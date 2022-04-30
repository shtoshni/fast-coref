import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
import math
from omegaconf import DictConfig
from typing import Dict, Tuple
from torch import Tensor

LOG2 = math.log(2)


class BaseMemory(nn.Module):
    """Base clustering module."""

    def __init__(self, config: DictConfig, span_emb_size: int, drop_module: nn.Module):
        super(BaseMemory, self).__init__()
        self.config = config

        self.mem_size = span_emb_size
        self.drop_module = drop_module

        if self.config.sim_func == "endpoint":
            num_embs = 2  # Span start, Span end
        else:
            num_embs = 3  # Span start, Span end, Hadamard product between the two

        self.mem_coref_mlp = MLP(
            num_embs * self.mem_size + config.num_feats * config.emb_size,
            config.mlp_size,
            1,
            drop_module=drop_module,
            num_hidden_layers=config.mlp_depth,
            bias=True,
        )

        if config.entity_rep == "learned_avg":
            # Parameter for updating the cluster representation
            self.alpha = MLP(
                2 * self.mem_size,
                config.mlp_size,
                1,
                num_hidden_layers=1,
                bias=True,
                drop_module=drop_module,
            )

        self.distance_embeddings = nn.Embedding(10, config.emb_size)
        self.counter_embeddings = nn.Embedding(10, config.emb_size)

    @property
    def device(self) -> torch.device:
        return next(self.mem_coref_mlp.parameters()).device

    def initialize_memory(
        self,
        mem: Tensor = None,
        ent_counter: Tensor = None,
        last_mention_start: Tensor = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Method to initialize the clusters and related bookkeeping variables."""
        # Check for unintialized memory
        if mem is None or ent_counter is None or last_mention_start is None:
            mem = torch.zeros(1, self.mem_size).to(self.device)
            ent_counter = torch.tensor([0.0]).to(self.device)
            last_mention_start = torch.zeros(1).long().to(self.device)

        return mem, ent_counter, last_mention_start

    @staticmethod
    def get_bucket(count: Tensor) -> Tensor:
        """Bucket distance and entity counters using the same logic."""

        logspace_idx = torch.floor(torch.log(count.float()) / LOG2).long() + 3
        use_identity = (count <= 4).long()
        combined_idx = use_identity * count + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, 0, 9)

    @staticmethod
    def get_distance_bucket(distances: Tensor) -> Tensor:
        return BaseMemory.get_bucket(distances)

    @staticmethod
    def get_counter_bucket(count: Tensor) -> Tensor:
        return BaseMemory.get_bucket(count)

    def get_distance_emb(self, distance: Tensor) -> Tensor:
        distance_tens = self.get_distance_bucket(distance)
        distance_embs = self.distance_embeddings(distance_tens)
        return distance_embs

    def get_counter_emb(self, ent_counter: Tensor) -> Tensor:
        counter_buckets = self.get_counter_bucket(ent_counter.long())
        counter_embs = self.counter_embeddings(counter_buckets)
        return counter_embs

    @staticmethod
    def get_coref_mask(ent_counter: Tensor) -> Tensor:
        """Mask for whether the cluster representation corresponds to any entity or not."""
        cell_mask = (ent_counter > 0.0).float()
        return cell_mask

    def get_feature_embs(
        self,
        ment_start: Tensor,
        last_mention_start: Tensor,
        ent_counter: Tensor,
        metadata: Dict,
    ) -> Tensor:
        distance_embs = self.get_distance_emb(ment_start - last_mention_start)
        counter_embs = self.get_counter_emb(ent_counter)

        feature_embs_list = [distance_embs, counter_embs]

        if "genre" in metadata:
            genre_emb = metadata["genre"]
            num_ents = distance_embs.shape[0]
            genre_emb = torch.unsqueeze(genre_emb, dim=0).repeat(num_ents, 1)
            feature_embs_list.append(genre_emb)

        feature_embs = self.drop_module(torch.cat(feature_embs_list, dim=-1))
        return feature_embs

    def get_coref_new_scores(
        self,
        ment_emb: Tensor,
        mem_vectors: Tensor,
        ent_counter: Tensor,
        feature_embs: Tensor,
    ) -> Tensor:
        """Calculate the coreference score with existing clusters.

        For creating a new cluster we use a dummy score of 0.
        This is a free variable and this idea is borrowed from Lee et al 2017

        Args:
                        ment_emb (d'): Mention representation
                        mem_vectors (M x d'): Cluster representations
                        ent_counter (M): Mention counter of clusters.
                        feature_embs (M x p): Embedding of features such as distance from last
                                        mention of the cluster.

        Returns:
                        coref_new_score (M + 1):
                                        Coref scores concatenated with the score of forming a new cluster.
        """

        # Repeat the query vector for comparison against all cells
        num_ents = mem_vectors.shape[0]
        rep_ment_emb = ment_emb.repeat(num_ents, 1)  # M x H

        # Coref Score
        if self.config.sim_func == "endpoint":
            pair_vec = torch.cat([mem_vectors, rep_ment_emb, feature_embs], dim=-1)
            pair_score = self.mem_coref_mlp(pair_vec)
        else:
            pair_vec = torch.cat(
                [mem_vectors, rep_ment_emb, mem_vectors * rep_ment_emb, feature_embs],
                dim=-1,
            )
            pair_score = self.mem_coref_mlp(pair_vec)

        coref_score = torch.squeeze(pair_score, dim=-1)  # M

        coref_new_mask = torch.cat(
            [self.get_coref_mask(ent_counter), torch.tensor([1.0], device=self.device)],
            dim=0,
        )
        # Use a dummy score of 0 for froming a new cluster
        coref_new_score = torch.cat(
            ([coref_score, torch.tensor([0.0], device=self.device)]), dim=0
        )
        coref_new_score = coref_new_score * coref_new_mask + (1 - coref_new_mask) * (
            -1e4
        )
        return coref_new_score

    @staticmethod
    def assign_cluster(coref_new_scores: Tensor) -> Tuple[int, str]:
        """Decode the action from argmax of clustering scores"""

        num_ents = coref_new_scores.shape[0] - 1
        pred_max_idx = torch.argmax(coref_new_scores).item()
        if pred_max_idx < num_ents:
            # Coref
            return pred_max_idx, "c"
        else:
            # New cluster
            return num_ents, "o"

    def coref_update(
        self, ment_emb: Tensor, mem_vectors: Tensor, cell_idx: int, ent_counter: Tensor
    ) -> Tensor:
        """Updates the cluster representation given the new mention representation."""

        if self.config.entity_rep == "learned_avg":
            alpha_wt = torch.sigmoid(
                self.alpha(torch.cat([mem_vectors[cell_idx], ment_emb], dim=0))
            )
            coref_vec = alpha_wt * mem_vectors[cell_idx] + (1 - alpha_wt) * ment_emb
        elif self.config.entity_rep == "max":
            coref_vec = torch.max(mem_vectors[cell_idx], ment_emb)
        else:
            cluster_count = ent_counter[cell_idx].item()
            coref_vec = (mem_vectors[cell_idx] * cluster_count + ment_emb) / (
                cluster_count + 1
            )

        return coref_vec
