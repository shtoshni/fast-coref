import torch
import torch.nn as nn

from auto_memory_model.controller.base_controller import BaseController
from pytorch_utils.modules import MLP


class Controller(BaseController):
    def __init__(self, mlp_size=1024, mlp_depth=1, max_span_width=30, top_span_ratio=0.4,
                 **kwargs):
        super(Controller, self).__init__(**kwargs)
        self.max_span_width = max_span_width
        self.mlp_size = mlp_size
        self.mlp_depth = mlp_depth
        self.top_span_ratio = top_span_ratio

        self.span_width_embeddings = nn.Embedding(self.max_span_width, 20)
        self.span_width_prior_embeddings = nn.Embedding(self.max_span_width, 20)
        self.mention_mlp = MLP(input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + 20,
                               hidden_size=self.mlp_size,
                               output_size=1, num_hidden_layers=self.mlp_depth, bias=True,
                               drop_module=self.drop_module)
        self.span_width_mlp = MLP(input_size=20, hidden_size=self.mlp_size,
                                  output_size=1, num_hidden_layers=1, bias=True,
                                  drop_module=self.drop_module)
        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def get_mention_width_scores(self, cand_starts, cand_ends):
        span_width_idx = cand_ends - cand_starts
        span_width_embs = self.span_width_prior_embeddings(span_width_idx)
        width_scores = torch.squeeze(self.span_width_mlp(span_width_embs), dim=-1)

        return width_scores

    def get_span_embeddings(self, encoded_doc, ment_starts, ment_ends):
        span_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]

        # Add span width embeddings
        span_width_indices = ment_ends - ment_starts
        span_width_embs = self.span_width_embeddings(span_width_indices)
        span_emb_list.append(span_width_embs)

        if self.ment_emb == 'attn':
            num_words = encoded_doc.shape[0]  # T
            num_c = ment_starts.shape[0]  # C
            doc_range = torch.unsqueeze(torch.arange(num_words), 0).repeat(num_c, 1).cuda()  # [C x T]
            ment_masks = ((doc_range >= torch.unsqueeze(ment_starts, dim=1)) &
                          (doc_range <= torch.unsqueeze(ment_ends, dim=1)))  # [C x T]
            word_attn = torch.squeeze(self.mention_attn(encoded_doc), dim=1)  # [T]
            mention_word_attn = nn.functional.softmax(
                (1 - ment_masks.float()) * (-1e10) + torch.unsqueeze(word_attn, dim=0), dim=1)  # [C x T]

            del ment_masks

            attention_term = torch.matmul(mention_word_attn, encoded_doc)  # C x H
            span_emb_list.append(attention_term)

        span_embs = torch.cat(span_emb_list, dim=-1)
        return span_embs

    def get_gold_mentions(self, clusters, num_words, flat_cand_mask):
        gold_ments = torch.zeros(num_words, self.max_span_width).cuda()
        for cluster in clusters:
            for (span_start, span_end) in cluster:
                span_width = span_end - span_start + 1
                if span_width <= self.max_span_width:
                    span_width_idx = span_width - 1
                    gold_ments[span_start, span_width_idx] = 1

        filt_gold_ments = gold_ments.reshape(-1)[flat_cand_mask].float()
        # assert(torch.sum(gold_ments) == torch.sum(filt_gold_ments))  # Filtering shouldn't remove gold mentions
        return filt_gold_ments

    def get_candidate_endpoints(self, encoded_doc, example):
        num_words = encoded_doc.shape[0]

        sent_map = torch.tensor(example["sentence_map"]).cuda()
        # num_words x max_span_width
        cand_starts = torch.unsqueeze(torch.arange(num_words), dim=1).repeat(1, self.max_span_width).cuda()
        cand_ends = cand_starts + torch.unsqueeze(torch.arange(self.max_span_width), dim=0).cuda()

        cand_start_sent_indices = sent_map[cand_starts]
        # Avoid getting sentence indices for cand_ends >= num_words
        corr_cand_ends = torch.min(cand_ends, torch.ones_like(cand_ends).cuda() * (num_words - 1))
        cand_end_sent_indices = sent_map[corr_cand_ends]

        # End before document ends & Same sentence
        constraint1 = (cand_ends < num_words)
        constraint2 = (cand_start_sent_indices == cand_end_sent_indices)
        cand_mask = constraint1 & constraint2
        flat_cand_mask = cand_mask.reshape(-1)

        # Filter and flatten the candidate end points
        filt_cand_starts = cand_starts.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        filt_cand_ends = cand_ends.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        return filt_cand_starts, filt_cand_ends, flat_cand_mask

    def forward(self, example):
        """
        Encode a batch of excerpts.
        """
        encoded_doc = self.doc_encoder(example)
        num_words = encoded_doc.shape[0]

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, example)

        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends)

        # Encoded doc not needed now
        del encoded_doc

        mention_logits = torch.squeeze(self.mention_mlp(span_embs), dim=-1)
        # Span embeddings not needed anymore
        del span_embs
        mention_logits += self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        filt_gold_mentions = self.get_gold_mentions(example["clusters"], num_words, flat_cand_mask)

        if self.training:
            mention_loss = self.mention_loss_fn(mention_logits, filt_gold_mentions)
            total_weight = filt_cand_starts.shape[0]

            loss = {'mention': mention_loss / total_weight}
            return loss

        else:
            pred_mention_probs = torch.sigmoid(mention_logits)
            # Calculate Recall
            k = int(self.top_span_ratio * num_words)
            topk_indices = torch.topk(mention_logits, k)[1]
            topk_indices_mask = torch.zeros_like(mention_logits).cuda()
            topk_indices_mask[topk_indices] = 1
            recall = torch.sum(filt_gold_mentions * topk_indices_mask).item()

            return pred_mention_probs, filt_gold_mentions, filt_cand_starts, filt_cand_ends, recall
