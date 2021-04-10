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

    def forward_training(self, instance):
        pred_mentions, mention_emb_list, mention_scores, train_vars = self.get_mention_embs(instance, topk=False)

        pred_mentions_list = pred_mentions.tolist()
        mention_to_cluster = get_mention_to_cluster_idx(instance["clusters"])
        gt_actions, _ = self.get_actions(pred_mentions_list, mention_to_cluster)

        metadata = {}
        if self.dataset == 'ontonotes':
            metadata = {'genre': self.get_genre_embedding(instance)}
        coref_new_list = self.memory_net.forward_training(pred_mentions, mention_emb_list, gt_actions, metadata)
        loss = {'total': train_vars['mention_loss'], 'entity': train_vars['mention_loss'].detach()}

        if len(coref_new_list) > 0:
            coref_loss = self.calculate_coref_loss(coref_new_list, gt_actions)
            loss['total'] += coref_loss
            loss['coref'] = coref_loss.detach()

        return loss

    def forward(self, instance, teacher_forcing=False):
        metadata = {}
        if self.dataset == 'ontonotes':
            metadata = {'genre': self.get_genre_embedding(instance)}

        action_list, pred_mentions_list, gt_actions = [], [], []
        last_memory, word_offset = None, 0
        cluster_to_cell, mention_to_cluster = {}, {}
        if "clusters" in instance:
            mention_to_cluster = get_mention_to_cluster_idx(instance["clusters"])

        for idx in range(0, len(instance["sentences"])):
            num_words = len(instance["sentences"][idx]) - 2
            cur_clusters = []
            for orig_cluster in instance["clusters"]:
                cluster = []
                for ment_start, ment_end in orig_cluster:
                    if ment_end >= word_offset and ment_start < word_offset + num_words:
                        cluster.append((ment_start - word_offset, ment_end - word_offset))

                if len(cluster):
                    cur_clusters.append(cluster)
            cur_example = {
                "padded_sent": instance["padded_sent"][idx],
                "sentence_map": instance["sentence_map"][word_offset: word_offset + num_words],
                "sent_len_list": [instance["sent_len_list"][idx]],
                "clusters": cur_clusters
            }

            cur_pred_mentions, cur_mention_emb_list = self.get_mention_embs(cur_example, topk=False)[:2]
            cur_pred_mentions = cur_pred_mentions + word_offset
            word_offset += num_words
            cur_pred_mentions_list = cur_pred_mentions.tolist()
            cur_gt_actions, cluster_to_cell = self.get_actions(pred_mentions_list, mention_to_cluster, cluster_to_cell)
            gt_actions.extend(cur_gt_actions)
            pred_mentions_list.extend(cur_pred_mentions_list)

            cur_action_list, last_memory = self.memory_net(
                cur_pred_mentions, cur_mention_emb_list, metadata, memory_init=last_memory)

            action_list.extend(cur_action_list)

        return action_list, pred_mentions_list, gt_actions
