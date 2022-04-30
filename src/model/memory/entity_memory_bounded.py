import torch
from model.memory import BaseMemory
from pytorch_utils.modules import MLP
import torch.nn as nn

from omegaconf import DictConfig
from typing import Dict, Tuple, List
from torch import Tensor


class EntityMemoryBounded(BaseMemory):
    """Module for clustering proposed mention spans using Entity-Ranking paradigm
    with bounded memory."""

    def __init__(
        self, config: DictConfig, span_emb_size: int, drop_module: nn.Module
    ) -> None:
        super(EntityMemoryBounded, self).__init__(config, span_emb_size, drop_module)
        self.max_ents: int = config.mem_type.max_ents
        self.bounded_mem_type: str = config.mem_type.name

        # Check if the memory is bounded i.e. there's a limit on the number of max entities
        self.fert_mlp = MLP(
            input_size=span_emb_size + config.num_feats * config.emb_size,
            bias=True,
            hidden_size=config.mlp_size,
            output_size=1,
            num_hidden_layers=config.mlp_depth,
            drop_module=drop_module,
        )

    def get_ment_feature_embs(self, metadata):
        # Bucket is 0 for both the embeddings
        distance_embs = self.distance_embeddings(torch.tensor(0, device=self.device))
        counter_embs = self.counter_embeddings(torch.tensor(0, device=self.device))

        feature_embs_list = [distance_embs, counter_embs]

        if "genre" in metadata:
            genre_emb = metadata["genre"]
            feature_embs_list.append(genre_emb)

        feature_embs = self.drop_module(torch.cat(feature_embs_list, dim=-1))
        return feature_embs

    def predict_new_or_ignore_learned(
        self,
        ment_emb: Tensor,
        mem_vectors: Tensor,
        feature_embs: Tensor,
        ment_feature_embs: Tensor,
    ) -> Tuple[Tensor, int, str]:
        """
        Predict whether a new entity is tracked or ignored.

        The key idea of this method is to predict fertility scores for different entity clusters
        and the current mention. The fertility score is supposed to reflect the number of
        remaining entities of a given entity cluster.
        """
        # Fertility Score
        mem_fert_input = torch.cat([mem_vectors, feature_embs], dim=-1)
        ment_fert_input = torch.unsqueeze(
            torch.cat([ment_emb, ment_feature_embs], dim=-1), dim=0
        )
        fert_input = torch.cat([mem_fert_input, ment_fert_input], dim=0)

        neg_fert_scores: Tensor = self.fert_mlp(fert_input)
        neg_fert_scores = torch.squeeze(neg_fert_scores, dim=-1)

        max_idx = int(torch.argmax(neg_fert_scores).item())
        if max_idx < self.max_ents:
            # The fertility of one of the entities currently being tracked is lower than the new entity.
            # We will overwrite this entity
            output = (
                neg_fert_scores,
                max_idx,
                "o",
            )
        else:
            # No space - The new entity is not "fertile" enough
            output = (
                neg_fert_scores,
                -1,
                "n",
            )

        return output

    def predict_new_or_ignore_lru(
        self,
        ment_emb: Tensor,
        mem_vectors: Tensor,
        feature_embs: Tensor,
        ment_feature_embs: Tensor,
        lru_list: List[int],
    ) -> Tuple[Tensor, int, str]:
        """
        Predict whether the new entity is tracked or ignored in the LRU scheme.

        The idea is to compare the fertility scores for the least recently seen entity cluster
        and the current entity cluster. The fertility scores are supposed to be ordered
        according to the number of mentions remaining in the entity cluster.
        """
        lru_cell = lru_list[0]
        mem_fert_input = torch.cat(
            [mem_vectors[lru_cell, :], feature_embs[lru_cell, :]], dim=0
        )
        ment_fert_input = torch.cat([ment_emb, ment_feature_embs], dim=-1)
        fert_input = torch.stack([mem_fert_input, ment_fert_input], dim=0)
        neg_fert_scores = torch.squeeze(self.fert_mlp(fert_input), dim=-1)
        output = (neg_fert_scores,)

        over_max_idx = torch.argmax(neg_fert_scores).item()
        if over_max_idx == 0:
            return output + (
                lru_cell,
                "o",
            )
        else:
            # No space - The new entity is not "fertile" enough
            return output + (
                -1,
                "n",
            )

    def forward_training(
        self,
        ment_boundaries: Tensor,
        mention_emb_list: List[Tensor],
        gt_actions: List[Tuple[int, str]],
        metadata: Dict,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Forward pass during coreference model training where we use teacher-forcing.

        Args:
                ment_boundaries: Mention boundaries of proposed mentions
                mention_emb_list: Embedding list of proposed mentions
                gt_actions: Ground truth clustering actions
                metadata: Metadata such as document genre

        Returns:
                coref_new_list: Logit scores for ground truth actions.
        """
        # Initialize memory
        first_overwrite, coref_new_list, new_ignore_list, = (
            True,
            [],
            [],
        )
        mem_vectors, ent_counter, last_mention_start = self.initialize_memory()
        lru_list = list(range(self.max_ents))

        for ment_idx, (ment_emb, (gt_cell_idx, gt_action_str)) in enumerate(
            zip(mention_emb_list, gt_actions)
        ):
            ment_start, ment_end = ment_boundaries[ment_idx]
            num_ents = 0 if mem_vectors is None else mem_vectors.shape[0]

            if first_overwrite:
                first_overwrite = False
                mem_vectors = torch.unsqueeze(ment_emb, dim=0)
                ent_counter = torch.tensor([1.0], device=self.device)
                last_mention_start = torch.tensor(
                    [ment_start], dtype=torch.long, device=self.device
                )
                continue
            else:
                feature_embs = self.get_feature_embs(
                    ment_start, last_mention_start, ent_counter, metadata
                )
                coref_new_scores = self.get_coref_new_scores(
                    ment_emb, mem_vectors, ent_counter, feature_embs
                )
                coref_new_list.append(coref_new_scores)

            # Check if memory has reached its limit
            if num_ents == self.max_ents and gt_action_str != "c":
                # Reached memory capacity
                if self.bounded_mem_type == "learned":
                    new_or_ignore_scores, _, _ = self.predict_new_or_ignore_learned(
                        ment_emb,
                        mem_vectors,
                        feature_embs,
                        self.get_ment_feature_embs(metadata),
                    )
                else:
                    # LRU memory scheeme
                    new_or_ignore_scores, _, _ = self.predict_new_or_ignore_lru(
                        ment_emb,
                        mem_vectors,
                        feature_embs,
                        self.get_ment_feature_embs(metadata),
                        lru_list,
                    )

                new_ignore_list.append(new_or_ignore_scores)

            # Teacher forcing
            action_str, cell_idx = gt_action_str, gt_cell_idx

            num_ents: int = int(torch.sum((ent_counter > 0).long()).item())
            cell_mask: Tensor = (
                torch.arange(start=0, end=num_ents, device=self.device)
                == torch.tensor(cell_idx)
            ).float()
            mask = torch.unsqueeze(cell_mask, dim=1)
            mask = mask.repeat(1, self.mem_size)

            if action_str == "c":
                coref_vec = self.coref_update(
                    ment_emb, mem_vectors, cell_idx, ent_counter
                )
                mem_vectors = mem_vectors * (1 - mask) + mask * coref_vec
                ent_counter[cell_idx] = ent_counter[cell_idx] + 1
                last_mention_start[cell_idx] = ment_start
            elif action_str == "o":
                if cell_idx == num_ents:
                    # Append the new vector, memory has not reached maximum capacity
                    mem_vectors = torch.cat(
                        [mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0
                    )
                    ent_counter = torch.cat(
                        [ent_counter, torch.tensor([1.0]).to(self.device)], dim=0
                    )
                    last_mention_start = torch.cat(
                        [last_mention_start, ment_start.unsqueeze(dim=0)], dim=0
                    )
                else:
                    # Replace the cell content tracking another entity
                    mem_vectors = mem_vectors * (1 - mask) + mask * ment_emb
                    last_mention_start[cell_idx] = ment_start
                    ent_counter[cell_idx] = 1.0

            if action_str in ["o", "c"]:
                # Coref or overwrite was chosen; place the cell_idx in use at the back of the list
                lru_list.remove(cell_idx)
                lru_list.append(cell_idx)

        return coref_new_list, new_ignore_list

    def forward(
        self,
        ment_boundaries: Tensor,
        mention_emb_list: Tensor,
        metadata: Dict,
        memory_init: Dict = None,
    ) -> Tuple[List[Tuple[int, str]], Dict]:
        """Forward pass for clustering entity mentions during inference/evaluation.

        Args:
         ment_boundaries: Start and end token indices for the proposed mentions.
         mention_emb_list: Embedding list of proposed mentions
         metadata: Metadata features such as document genre embedding
         memory_init: Initializer for memory. For streaming coreference, we can pass the previous
                  memory state via this dictionary

        Returns:
                pred_actions: List of predicted clustering actions.
                mem_state: Current memory state.
        """
        # Initialize memory
        if memory_init is not None:
            mem_vectors, ent_counter, last_mention_start = self.initialize_memory(
                **memory_init
            )
            lru_list = memory_init["lru_list"]
        else:
            mem_vectors, ent_counter, last_mention_start = self.initialize_memory()
            lru_list = list(range(self.max_ents))

        pred_actions = []  # argmax actions

        # Boolean to track if we have started tracking any entities
        # This value can be false if when processing subsequent chunks of a long document
        first_overwrite: bool = True if torch.sum(ent_counter) == 0 else False

        for ment_idx, ment_emb in enumerate(mention_emb_list):
            ment_start, ment_end = ment_boundaries[ment_idx]
            feature_embs = self.get_feature_embs(
                ment_start, last_mention_start, ent_counter, metadata
            )
            num_ents = 0 if mem_vectors is None else mem_vectors.shape[0]

            if first_overwrite:
                # First mention is always an overwrite
                pred_cell_idx, pred_action_str = 0, "o"
            else:
                # Predict whether the mention is coreferent with existing clusters or a new cluster
                coref_new_scores = self.get_coref_new_scores(
                    ment_emb, mem_vectors, ent_counter, feature_embs
                )
                pred_cell_idx, pred_action_str = self.assign_cluster(coref_new_scores)

                if num_ents == self.max_ents and pred_action_str != "c":
                    # Reached memory capacity
                    if self.bounded_mem_type == "learned":
                        (
                            new_or_ignore_scores,
                            pred_cell_idx,
                            pred_action_str,
                        ) = self.predict_new_or_ignore_learned(
                            ment_emb,
                            mem_vectors,
                            feature_embs,
                            self.get_ment_feature_embs(metadata),
                        )
                    else:
                        # LRU memory scheme
                        (
                            new_or_ignore_scores,
                            pred_cell_idx,
                            pred_action_str,
                        ) = self.predict_new_or_ignore_lru(
                            ment_emb,
                            mem_vectors,
                            feature_embs,
                            self.get_ment_feature_embs(metadata),
                            lru_list,
                        )

            pred_actions.append((pred_cell_idx, pred_action_str))

            if first_overwrite:
                first_overwrite = False
                # We start with a single empty memory cell
                mem_vectors = torch.unsqueeze(ment_emb, dim=0)
                ent_counter = torch.tensor([1.0], device=self.device)
                last_mention_start[0] = ment_start
            else:
                if pred_action_str == "c":
                    # Perform coreference update on the cluster referenced by pred_cell_idx
                    coref_vec = self.coref_update(
                        ment_emb, mem_vectors, pred_cell_idx, ent_counter
                    )
                    mem_vectors[pred_cell_idx] = coref_vec
                    ent_counter[pred_cell_idx] = ent_counter[pred_cell_idx] + 1
                    last_mention_start[pred_cell_idx] = ment_start

                elif pred_action_str == "o":
                    # Append the new entity to the entity cluster array
                    if pred_cell_idx == num_ents:
                        mem_vectors = torch.cat(
                            [mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0
                        )
                        ent_counter = torch.cat(
                            [ent_counter, torch.tensor([1.0]).to(self.device)], dim=0
                        )
                        last_mention_start = torch.cat(
                            [last_mention_start, ment_start.unsqueeze(dim=0)], dim=0
                        )
                    else:
                        # Replace the cell content tracking another entity
                        mem_vectors[pred_cell_idx] = ment_emb
                        last_mention_start[pred_cell_idx] = ment_start
                        ent_counter[pred_cell_idx] = 1.0

            if pred_action_str in ["o", "c"]:
                # Coref or overwrite was chosen; place the cell_idx in use at the back of the list
                lru_list.remove(pred_cell_idx)
                lru_list.append(pred_cell_idx)

        mem_state = {
            "mem": mem_vectors,
            "ent_counter": ent_counter,
            "last_mention_start": last_mention_start,
            "lru_list": lru_list,
        }
        return pred_actions, mem_state
