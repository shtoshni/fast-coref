import torch
import torch.nn as nn
import numpy as np

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

    def get_actions(self, example, pred_mentions, rand_fl_list, follow_gt, sample_invalid):
        if "clusters" in example:
            gt_clusters = example["clusters"]
            if self.mem_type == 'unbounded':
                return get_actions_unbounded(pred_mentions, gt_clusters, rand_fl_list, follow_gt, sample_invalid)
            elif self.mem_type == 'learned':
                return get_actions_learned_bounded(pred_mentions, gt_clusters, max_ents=self.max_ents)
            elif self.mem_type == 'lru':
                return get_actions_lru(pred_mentions, gt_clusters, max_ents=self.max_ents)
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

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        pred_mentions, mention_emb_list, mention_score_list = self.get_mention_embs(example)

        follow_gt = self.training or teacher_forcing
        rand_fl_list = np.random.random(len(mention_emb_list))
        if teacher_forcing:
            rand_fl_list = np.zeros_like(rand_fl_list)

        gt_actions = self.get_actions(example, pred_mentions, rand_fl_list, follow_gt, self.sample_invalid)

        metadata = {}
        if self.dataset == 'ontonotes':
            metadata = {'genre': self.get_genre_embedding(example)}

        entity_or_not_list, coref_new_list, new_ignore_list, action_list = self.memory_net(
            pred_mentions, mention_emb_list, mention_score_list, gt_actions, metadata, rand_fl_list,
            teacher_forcing=teacher_forcing)
        loss = {'total': None}
        if follow_gt:
            if len(entity_or_not_list) > 0:
                entity_invalid_tens = torch.stack(entity_or_not_list, dim=0)
                entity_or_not_indices = self.entity_or_not_entity_gt(action_list)
                entity_loss = torch.sum(self.loss_fn(entity_invalid_tens, entity_or_not_indices))
                loss['entity'] = entity_loss
                loss['total'] = loss['entity']
                if len(coref_new_list) > 0:
                    coref_loss = self.calculate_coref_loss(coref_new_list, gt_actions)
                    loss['coref'] = coref_loss
                    loss['total'] += loss['coref']

                    # Calculate new-ignore loss
                    if self.is_mem_bounded and len(new_ignore_list) > 0:
                        new_ignore_tens = torch.stack(new_ignore_list, dim=0)
                        new_ignore_indices = self.new_ignore_tuple_to_idx(gt_actions)
                        # ignore_loss = torch.sum(self.label_smoothing_loss_fn(new_ignore_tens, new_ignore_indices))
                        ignore_loss = torch.sum(self.label_smoothing_loss_fn(
                            new_ignore_tens, torch.unsqueeze(new_ignore_indices, dim=1)))
                        loss['ignore'] = ignore_loss
                        loss['total'] += loss['ignore']
            return loss, action_list, pred_mentions, gt_actions
        else:
            mention_scores = [mention_score.item() for mention_score in mention_score_list]
            return 0.0, action_list, pred_mentions, mention_scores, gt_actions
