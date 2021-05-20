import torch

from auto_memory_model.memory import MemoryPredInvalid
from auto_memory_model.controller import BaseController
from auto_memory_model.controller.utils_action import *
from pytorch_utils.label_smoothing import LabelSmoothingLoss


class ControllerPredInvalid(BaseController):
    def __init__(self, max_ents=None, mem_type='unbounded', **kwargs):
        super(ControllerPredInvalid, self).__init__(**kwargs)
        self.mem_type = mem_type
        if mem_type != 'unbounded':
            self.max_ents = max_ents
            self.is_mem_bounded = True
        else:
            self.max_ents = None
            self.is_mem_bounded = False

        self.memory_net = MemoryPredInvalid(
            mem_type=mem_type, max_ents=self.max_ents,
            hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
            drop_module=self.drop_module, num_feats=self.num_feats, **kwargs).to(self.device)

        self.label_smoothing_loss_fn = LabelSmoothingLoss(smoothing=self.label_smoothing_wt, dim=1)

    @staticmethod
    def get_actions(pred_mentions, instance):
        if "clusters" in instance:
            mention_to_cluster = get_mention_to_cluster_idx(instance["clusters"])
            return get_actions_unbounded_fast(pred_mentions, mention_to_cluster)
        else:
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

    def forward_training(self, instance):
        pred_mentions, mention_emb_list, _, train_vars = self.get_mention_embs(instance)

        pred_mentions_list = pred_mentions.tolist()
        gt_actions = self.get_actions(pred_mentions_list, instance)

        metadata = self.get_metadata(instance)

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

    def forward(self, instance, teacher_forcing=False):
        metadata = self.get_metadata(instance)

        action_list, pred_mentions_list, gt_actions, mention_scores = [], [], [], []
        last_memory, token_offset = None, 0

        for idx in range(0, len(instance["sentences"])):
            num_tokens = len(instance["sentences"][idx])

            clusters = []
            for orig_cluster in instance.get("clusters", []):
                cluster = []
                for mention in orig_cluster:
                    ment_start, ment_end = mention[:2]
                    if ment_start >= token_offset and ment_end < (token_offset + num_tokens):
                        cluster.append((ment_start - token_offset, ment_end - token_offset))

                if len(cluster):
                    clusters.append(cluster)

            cur_example = {
                "tensorized_sent": instance["tensorized_sent"][idx],
                "sentence_map": instance["sentence_map"][token_offset: token_offset + num_tokens],
                "subtoken_map": instance["subtoken_map"][token_offset: token_offset + num_tokens],
                "sent_len_list": [instance["sent_len_list"][idx]],
                "clusters": clusters,
            }

            # Pass along other metadata
            for key in instance:
                if key not in cur_example:
                    cur_example[key] = instance[key]

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

        gt_actions = self.get_actions(pred_mentions_list, instance)
        return action_list, pred_mentions_list, gt_actions, mention_scores
