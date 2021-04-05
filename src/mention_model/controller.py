import torch

from auto_memory_model.controller.base_controller import BaseController


class Controller(BaseController):
    def __init__(self, **kwargs):
        super(Controller, self).__init__(**kwargs)

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        encoded_doc = self.doc_encoder(example)
        num_words = encoded_doc.shape[0]

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(
            encoded_doc, example)

        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends)

        mention_logits = torch.squeeze(self.mention_mlp(span_embs), dim=-1)
        # Span embeddings not needed anymore
        mention_logits += self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        filt_gold_mentions = self.get_gold_mentions(example["clusters"], num_words, flat_cand_mask)
        # print(torch.sum(filt_gold_mentions))

        if self.training:
            mention_loss = self.mention_loss_fn(mention_logits, filt_gold_mentions)
            total_weight = filt_cand_starts.shape[0]

            loss = {'mention': mention_loss / total_weight}
            return loss,
        else:
            # Calculate Recall
            k = int(self.top_span_ratio * num_words)
            topk_indices = torch.topk(mention_logits, k)[1]
            topk_indices_mask = torch.zeros_like(mention_logits, device=mention_logits.device)
            topk_indices_mask[topk_indices] = 1
            recall = torch.sum(filt_gold_mentions * topk_indices_mask).item()

            topk_starts = filt_cand_starts[topk_indices]
            topk_ends = filt_cand_ends[topk_indices]
            topk_scores = torch.sigmoid(mention_logits[topk_indices])

            # Sort the mentions by (start) and tiebreak with (end)
            sort_scores = topk_starts + 1e-5 * topk_ends
            _, sorted_indices = torch.sort(sort_scores, 0)

            pred_mentions = list(zip(topk_starts[sorted_indices].tolist(), topk_ends[sorted_indices].tolist()))
            mention_scores = topk_scores[sorted_indices].tolist()
            return pred_mentions, mention_scores, recall, torch.sum(filt_gold_mentions).item()
