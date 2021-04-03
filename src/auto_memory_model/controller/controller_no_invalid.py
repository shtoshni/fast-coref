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
        encoded_doc = self.doc_encoder(example, max_training_segments=max_training_segments)
        pred_starts, pred_ends, pred_scores = self.get_pred_mentions(example, encoded_doc, topk=True)

        # Sort the predicted mentions
        pred_mentions = list(zip(pred_starts.tolist(), pred_ends.tolist()))
        mention_score_list = torch.unbind(torch.unsqueeze(pred_scores, dim=1))

        mention_embs = self.get_span_embeddings(encoded_doc, pred_starts, pred_ends)
        mention_emb_list = torch.unbind(mention_embs, dim=0)

        metadata = {}
        if self.dataset == 'ontonotes':
            metadata = {'genre': self.get_genre_embedding(example)}

        follow_gt = self.training or teacher_forcing
        rand_fl_list = np.random.random(len(mention_emb_list))
        if teacher_forcing:
            rand_fl_list = np.zeros_like(rand_fl_list)

        if "clusters" in example:
            gt_actions = get_actions_unbounded(
                pred_mentions, example["clusters"], rand_fl_list, follow_gt, self.sample_invalid)
        else:
            gt_actions = [(-1, 'i')] * len(pred_mentions)
        action_prob_list, action_list = self.memory_net(
            mention_emb_list, mention_score_list, gt_actions, metadata, teacher_forcing=teacher_forcing)

        loss = {'total': None}
        if follow_gt:
            if len(action_prob_list) > 0:
                coref_new_prob_list = action_prob_list
                loss = {}
                coref_loss = self.calculate_coref_loss(coref_new_prob_list, gt_actions)
                loss['coref'] = coref_loss
                loss['total'] = loss['coref']

            return loss, action_list, pred_mentions, gt_actions
        else:
            mention_scores = [mention_score.item() for mention_score in mention_score_list]
            return 0.0, action_list, pred_mentions, mention_scores, gt_actions
