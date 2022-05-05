import torch
import torch.nn as nn

from model.mention_proposal import MentionProposalModule
from model.utils import get_gt_actions
from model.memory.entity_memory import EntityMemory
from model.memory.entity_memory_bounded import EntityMemoryBounded

from typing import Dict, List, Tuple
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedTokenizerFast

import logging
import random

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


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
            self.config, train_config, drop_module=self.drop_module
        )

        # Clustering module
        span_emb_size: int = self.mention_proposer.span_emb_size
        # Use of genre feature in clustering or not
        if self.config.metadata_params.use_genre_feature:
            self.config.memory.num_feats = 3

        self.mem_type = self.config.memory.mem_type.name

        if self.mem_type == "unbounded":
            self.memory_net = EntityMemory(
                config=self.config.memory,
                span_emb_size=span_emb_size,
                drop_module=self.drop_module,
            )
        else:
            self.memory_net = EntityMemoryBounded(
                config=self.config.memory,
                span_emb_size=span_emb_size,
                drop_module=self.drop_module,
            )

        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.train_config.label_smoothing_wt
        )

        if self.config.metadata_params.use_genre_feature:
            self.genre_embeddings = nn.Embedding(
                num_embeddings=len(self.config.metadata_params.genres),
                embedding_dim=self.config.mention_params.emb_size,
            )

    @property
    def device(self) -> torch.device:
        return self.mention_proposer.device

    def get_params(self, named=False) -> Tuple[List, List]:
        """Returns a tuple of document encoder parameters and rest of the model params."""

        encoder_params, mem_params = [], []
        for name, param in self.named_parameters():
            elem = (name, param) if named else param
            if "doc_encoder" in name:
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
                doc_class_idx = meta_params.genres.index(doc_class)
            else:
                doc_class_idx = meta_params.genres.index(
                    meta_params.default_genre
                )  # Default genre

            return {
                "genre": self.genre_embeddings(
                    torch.tensor(doc_class_idx, device=self.device)
                )
            }
        else:
            return {}

    def calculate_new_ignore_loss(
        self, new_ignore_list: List, action_tuple_list: List[Tuple[int, str]]
    ) -> Tensor:
        ent_counter = 0
        max_ents = self.config.memory.mem_type.max_ents

        ignore_loss = torch.tensor(0.0, device=self.device)

        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            if action_str == "c":
                continue
            else:
                # New entity
                ent_counter += 1

                if action_str == "o":
                    if self.mem_type == "lru":
                        gt_idx = 0
                    else:
                        gt_idx = cell_idx
                else:
                    # No space
                    if self.mem_type == "lru":
                        gt_idx = 1
                    else:
                        gt_idx = max_ents

                if ent_counter > max_ents:
                    # Reached memory capacity
                    index = ent_counter - max_ents - 1
                    target = torch.tensor([gt_idx], device=self.device)
                    ignore_loss += self.loss_fn(
                        torch.unsqueeze(new_ignore_list[index], dim=0), target
                    )

        return ignore_loss

    def calculate_coref_loss(
        self, action_prob_list: List, action_tuple_list: List[Tuple[int, str]]
    ) -> Tensor:
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
        coref_loss = torch.tensor(0.0, device=self.device)

        max_ents = self.config.memory.mem_type.max_ents

        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            if action_str == "c":
                # Coreference with clusters currently tracked
                gt_idx = cell_idx

            elif action_str == "o":
                # Overwrite - New cluster
                gt_idx = num_ents
                if max_ents is None or num_ents < max_ents:
                    num_ents += 1

                if num_ents == 1:  # The first ent is always overwritten - No loss there
                    continue
            else:
                continue

            target = torch.tensor([gt_idx], device=self.device)
            coref_loss += self.loss_fn(
                torch.unsqueeze(action_prob_list[counter], dim=0), target
            )
            counter += 1

        return coref_loss

    @staticmethod
    def get_filtered_clusters(
        clusters, init_token_offset, final_token_offset, with_offset=True
    ):
        """Filter clusters from a document given the token offsets."""
        filt_clusters = []
        for orig_cluster in clusters:
            cluster = []
            for ment_start, ment_end in orig_cluster:
                if ment_start >= init_token_offset and ment_end < final_token_offset:
                    if with_offset:
                        cluster.append((ment_start, ment_end))
                    else:
                        cluster.append(
                            (
                                ment_start - init_token_offset,
                                ment_end - init_token_offset,
                            )
                        )
            if len(cluster):
                filt_clusters.append(cluster)

        return filt_clusters

    def forward_training(self, document: Dict) -> Dict:
        """Forward pass for training.

        Args:
                document: The tensorized document.

        Returns:
                loss_dict (Dict): Loss dictionary containing the losses of different stages of the model.
        """

        # Truncate document for training
        max_training_segments = self.train_config.get("max_training_segments", None)
        num_segments = len(document["sentences"])
        if max_training_segments is None:
            seg_range = [0, num_segments]
        else:
            if num_segments > max_training_segments:
                start_seg = random.randint(0, num_segments - max_training_segments)
                seg_range = [start_seg, start_seg + max_training_segments]
            else:
                seg_range = [0, num_segments]

        # Initialize lists to track all the mentions predicted across the chunks
        pred_mentions_list, mention_emb_list = [], []
        init_token_offset = sum(
            [len(document["sentences"][idx]) for idx in range(0, seg_range[0])]
        )
        token_offset = init_token_offset

        # logger.info(f"Token offset: {token_offset}, # of sentences: {num_segments}")

        # Metadata such as document genre can be used by model for clustering
        metadata = self.get_metadata(document)

        # Initialize the mention loss
        ment_loss = None
        # Bounded memory loss
        ignore_loss = None

        # Step 1: Predict all the mentions
        for idx in range(seg_range[0], seg_range[1]):
            num_tokens = len(document["sentences"][idx])

            cur_doc_slice = {
                "tensorized_sent": document["tensorized_sent"][idx],
                "sentence_map": document["sentence_map"][
                    token_offset : token_offset + num_tokens
                ],
                "subtoken_map": document["subtoken_map"][
                    token_offset : token_offset + num_tokens
                ],
                "sent_len_list": [document["sent_len_list"][idx]],
                "clusters": self.get_filtered_clusters(
                    document["clusters"],
                    token_offset,
                    token_offset + num_tokens,
                    with_offset=False,
                ),
                "doc_key": document["doc_key"],
            }

            proposer_output_dict = self.mention_proposer(cur_doc_slice)
            if proposer_output_dict.get("ments", None) is None:
                token_offset += num_tokens
                continue

            # Add the document offset to mentions predicted for the current chunk
            cur_pred_mentions = proposer_output_dict.get("ments") + token_offset

            pred_mentions_list.extend(cur_pred_mentions.tolist())
            mention_emb_list.extend(proposer_output_dict["ment_emb_list"])
            if "ment_loss" in proposer_output_dict:
                if ment_loss is None:
                    ment_loss = proposer_output_dict["ment_loss"]
                else:
                    ment_loss += proposer_output_dict["ment_loss"]

            # Update the document offset for next iteration
            token_offset += num_tokens

        # Step 2: Perform clustering
        # Get clusters part of the truncated document
        truncated_document_clusters = {
            "clusters": self.get_filtered_clusters(
                document["clusters"], init_token_offset, token_offset
            )
        }
        # Get ground truth clustering mentions
        gt_actions: List[Tuple[int, str]] = get_gt_actions(
            pred_mentions_list, truncated_document_clusters, self.config.memory.mem_type
        )

        pred_mentions = torch.tensor(pred_mentions_list, device=self.device)

        if self.mem_type == "unbounded":
            coref_new_list = self.memory_net.forward_training(
                pred_mentions, mention_emb_list, gt_actions, metadata
            )
        else:
            coref_new_list, new_ignore_list = self.memory_net.forward_training(
                pred_mentions, mention_emb_list, gt_actions, metadata
            )

            if len(new_ignore_list):
                ignore_loss = self.calculate_new_ignore_loss(
                    new_ignore_list, gt_actions
                )

        # Consolidate different losses in one dictionary
        if ment_loss is not None:
            loss_dict = {"total": ment_loss, "entity": ment_loss}
        else:
            loss_dict = {"total": 0.0}

        if len(coref_new_list) > 0:
            coref_loss = self.calculate_coref_loss(coref_new_list, gt_actions)
            loss_dict["total"] = loss_dict["total"] + coref_loss
            loss_dict["coref"] = coref_loss
            if ignore_loss is not None:
                loss_dict["bounded"] = ignore_loss
                loss_dict["total"] = loss_dict["total"] + ignore_loss
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
        pred_mentions_list, mention_scores, pred_actions = [], [], []
        # Initialize entity clusters and current document token offset
        entity_cluster_states, token_offset = None, 0

        metadata = self.get_metadata(document)

        for idx in range(0, len(document["sentences"])):
            num_tokens = len(document["sentences"][idx])

            cur_example = {
                "tensorized_sent": document["tensorized_sent"][idx],
                "sentence_map": document["sentence_map"][
                    token_offset : token_offset + num_tokens
                ],
                "subtoken_map": document["subtoken_map"][
                    token_offset : token_offset + num_tokens
                ],
                "sent_len_list": [document["sent_len_list"][idx]],
            }

            # Pass along other metadata
            for key in document:
                if key not in cur_example:
                    cur_example[key] = document[key]

            proposer_output_dict = self.mention_proposer(cur_example)
            if proposer_output_dict.get("ments", None) is None:
                token_offset += num_tokens
                continue

            # Add the document offset to mentions predicted for the current chunk
            # It's important to add the offset before clustering because features like
            # number of tokens between the last mention of the cluster and the current mention
            # will be affected if the current token indices of the mention are not supplied.
            cur_pred_mentions = proposer_output_dict.get("ments") + token_offset

            # Update the document offset for next iteration
            token_offset += num_tokens

            pred_mentions_list.extend(cur_pred_mentions.tolist())
            mention_scores.extend(proposer_output_dict["ment_scores"].tolist())

            # Pass along entity clusters from previous chunks while processing next chunks
            cur_pred_actions, entity_cluster_states = self.memory_net(
                cur_pred_mentions,
                proposer_output_dict["ment_emb_list"],
                metadata,
                memory_init=entity_cluster_states,
            )
            pred_actions.extend(cur_pred_actions)

        gt_actions = get_gt_actions(
            pred_mentions_list, document, self.config.memory.mem_type
        )  # Useful for oracle calcs
        return pred_mentions_list, mention_scores, gt_actions, pred_actions
