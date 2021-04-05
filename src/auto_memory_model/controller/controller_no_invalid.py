import torch
import numpy as np

from auto_memory_model.memory import MemoryNoInvalid
from auto_memory_model.controller import BaseController
from auto_memory_model.controller.utils_action import get_actions_unbounded


class ControllerNoInvalid(BaseController):
    def __init__(self, **kwargs):
        super(ControllerNoInvalid, self).__init__(**kwargs)

        self.memory_net = MemoryNoInvalid(
            hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
            drop_module=self.drop_module, **kwargs)

    def forward(self, example, teacher_forcing=False, max_training_segments=None):
        """
        Encode a batch of excerpts.
        """
        pred_mentions, mention_emb_list, mention_scores, mention_loss = \
            self.get_mention_embs(example, topk=True)

        follow_gt = self.training or teacher_forcing
        pred_mentions_list = pred_mentions.tolist()

        metadata = {}
        if self.dataset == 'ontonotes':
            metadata = {'genre': self.get_genre_embedding(example)}

        if "clusters" in example:
            gt_actions = get_actions_unbounded(pred_mentions_list, example["clusters"])
        else:
            gt_actions = [(-1, 'i')] * len(pred_mentions_list)

        action_prob_list, action_list = self.memory_net(
            mention_emb_list, mention_scores, gt_actions, metadata, teacher_forcing=teacher_forcing)

        loss = {'total': None}
        if follow_gt:
            if mention_loss is not None:
                loss['entity'] = mention_loss
                loss['total'] = loss['entity']
            if len(action_prob_list) > 0:
                coref_new_prob_list = action_prob_list
                loss = {}
                coref_loss = self.calculate_coref_loss(coref_new_prob_list, gt_actions)
                loss['coref'] = coref_loss
                loss['total'] = loss['coref']

            return loss, action_list, pred_mentions, gt_actions
        else:
            return action_list, pred_mentions_list, mention_scores, gt_actions
