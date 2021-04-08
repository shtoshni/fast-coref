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
            drop_module=self.drop_module, **kwargs)

        self.label_smoothing_loss_fn = LabelSmoothingLoss(smoothing=self.label_smoothing_wt, dim=1)

    @staticmethod
    def get_actions(pred_mentions, mention_to_cluster, cluster_to_cell=None):
        return get_actions_unbounded_fast(pred_mentions, mention_to_cluster, cluster_to_cell=cluster_to_cell)

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

    def forward(self, example, teacher_forcing=False):
        metadata = {}
        if self.dataset == 'ontonotes':
            metadata = {'genre': self.get_genre_embedding(example)}

        follow_gt = self.training or teacher_forcing
        cell_to_cluster, cluster_to_cell = {}, {}
        coref_new_list, new_ignore_list, action_list, pred_mentions_list, gt_actions = [], [], [], [], []
        loss = {'total': None}
        mention_loss = None
        last_memory = None

        if "clusters" in example:
            mention_to_cluster = get_mention_to_cluster_idx(example["clusters"])

        word_offset = 0
        for idx in range(0, len(example["sentences"])):
            num_words = len(example["sentences"][idx]) - 2
            cur_clusters = []
            for orig_cluster in example["clusters"]:
                cluster = []
                for ment_start, ment_end in orig_cluster:
                    if ment_end >= word_offset and ment_start < word_offset + num_words:
                        cluster.append((ment_start - word_offset, ment_end - word_offset))

                if len(cluster):
                    cur_clusters.append(cluster)
            cur_example = {
                "padded_sent": example["padded_sent"][idx],
                "sentence_map": example["sentence_map"][word_offset: word_offset + num_words],
                "sent_len_list": [example["sent_len_list"][idx]],
                "clusters": cur_clusters
            }

            cur_pred_mentions, cur_mention_emb_list, _, cur_mention_loss = \
                self.get_mention_embs(cur_example, topk=False)
            if mention_loss is None:
                if cur_mention_loss is not None:
                    mention_loss = cur_mention_loss
            else:
                mention_loss += cur_mention_loss

            cur_pred_mentions = cur_pred_mentions + word_offset
            word_offset += num_words
            pred_mentions_list.extend(cur_pred_mentions.tolist())

            if "clusters" in example:
                cur_gt_actions, cluster_to_cell = self.get_actions(
                    cur_pred_mentions, mention_to_cluster, cluster_to_cell=cluster_to_cell)
            else:
                cur_gt_actions = [(-1, 'i')] * len(cur_pred_mentions)

            gt_actions.extend(cur_gt_actions)

            cur_coref_new_list, cur_new_ignore_list, cur_action_list, last_memory = self.memory_net(
                cur_pred_mentions, cur_mention_emb_list, cur_gt_actions, metadata, memory_init=last_memory,
                teacher_forcing=teacher_forcing)

            coref_new_list.extend(cur_coref_new_list)
            new_ignore_list.extend(cur_new_ignore_list)
            action_list.extend(cur_action_list)

        if follow_gt:
            if mention_loss is not None:
                loss['entity'] = mention_loss
                loss['total'] = loss['entity']
            if len(coref_new_list) > 0:
                coref_loss = self.calculate_coref_loss(coref_new_list, gt_actions)
                loss['coref'] = coref_loss
                loss['total'] += loss['coref']

                # Calculate new-ignore loss
                if self.is_mem_bounded and len(new_ignore_list) > 0:
                    new_ignore_tens = torch.stack(new_ignore_list, dim=0)
                    new_ignore_indices = self.new_ignore_tuple_to_idx(gt_actions)
                    ignore_loss = torch.sum(self.label_smoothing_loss_fn(
                        new_ignore_tens, torch.unsqueeze(new_ignore_indices, dim=1)))
                    loss['ignore'] = ignore_loss
                    loss['total'] += loss['ignore']
            return loss
        else:
            return action_list, pred_mentions_list, gt_actions


