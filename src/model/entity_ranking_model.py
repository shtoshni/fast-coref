import torch
import torch.nn as nn

from model.mention_proposal_module import MentionProposalModule
from model.utils import get_mention_to_cluster_idx, get_actions_unbounded_fast, \
    get_actions_lru, get_actions_learned
from pytorch_utils.label_smoothing import LabelSmoothingLoss


class EntityRankingModel(nn.Module):
    def __init__(self, model_config, train_config):
        super(EntityRankingModel, self).__init__()
        self.mention_proposer = MentionProposalModule(model_config, train_config)

        # self.memory_net = MemoryPredInvalid(
        #     mem_type=mem_type, max_ents=self.max_ents,
        #     hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
        #     drop_module=self.drop_module, num_feats=self.num_feats, **kwargs).to(self.device)

    @staticmethod
    def get_gt_actions(pred_mentions, document):
        if "clusters" in document:
            mention_to_cluster = get_mention_to_cluster_idx(document["clusters"])
            return get_actions_unbounded_fast(pred_mentions, mention_to_cluster)
        else:
            # Don't have ground truth actions; generate dummy actions
            return [(-1, 'i')] * len(pred_mentions)

    def new_ignore_tuple_to_idx(self, action_tuple_list):
        action_indices = []

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
                    action_indices.append(self.max_ents)

        # The first max_ents are all overwrites - We skip that part
        if len(action_indices) > self.max_ents:
            action_indices = action_indices[self.max_ents:]
            action_indices = torch.tensor(action_indices, device=self.device)
            return action_indices
        else:
            return []

    def calculate_coref_loss(self, action_prob_list, action_tuple_list):
        '''Calculates the coreference loss given the action probability list and ground truth actions.'''
        num_ents, counter = 0, 0
        coref_loss = 0.0

        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            if action_str == 'c':
                # Coreference with clusters currently tracked
                gt_idx = cell_idx

            elif action_str == 'o':
                # Overwrite - New cluster
                gt_idx = num_ents
                if self.max_ents is None or num_ents < self.max_ents:
                    num_ents += 1

                if num_ents == 1:
                    # The first ent is always overwritten - No loss there
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
        pred_mentions, mention_emb_list, _, train_vars = self.get_mention_embs(document)

        pred_mentions_list = pred_mentions.tolist()
        gt_actions = self.get_gt_actions(pred_mentions_list, document)

        metadata = self.get_metadata(document)

        coref_new_list = self.memory_net.forward_training(pred_mentions, mention_emb_list, gt_actions, metadata)
        if 'mention_loss' in train_vars:
            loss = {'total': train_vars['mention_loss'], 'entity': train_vars['mention_loss']}

            if len(coref_new_list) > 0:
                coref_loss = self.calculate_coref_loss(coref_new_list, gt_actions)
                loss['total'] += coref_loss
                loss['coref'] = coref_loss
        else:
            loss = {'total': None}

        return loss

    def forward(self, document):
        metadata = self.get_metadata(document)

        action_list, pred_mentions_list, gt_actions, mention_scores = [], [], [], []
        last_memory, token_offset = None, 0

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

            cur_pred_mentions, cur_mention_emb_list, cur_mention_scores = self.get_mention_embs(cur_example)[:3]
            if cur_pred_mentions is None:
                continue
            cur_pred_mentions = cur_pred_mentions + token_offset
            token_offset += num_tokens

            pred_mentions_list.extend(cur_pred_mentions.tolist())
            mention_scores.extend(cur_mention_scores.tolist())

            cur_action_list, last_memory = self.memory_net(
                cur_pred_mentions, cur_mention_emb_list, metadata, memory_init=last_memory)
            action_list.extend(cur_action_list)

        gt_actions = self.get_actions(pred_mentions_list, document)
        return action_list, pred_mentions_list, gt_actions, mention_scores
