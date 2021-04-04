import torch
import torch.nn as nn

from document_encoder.independent import IndependentDocEncoder
from document_encoder.overlap import OverlapDocEncoder
from pytorch_utils.modules import MLP
from pytorch_utils.label_smoothing import LabelSmoothingLoss


class BaseController(nn.Module):
    def __init__(self,
                 dropout_rate=0.5, max_span_width=20, top_span_ratio=0.4,
                 ment_emb='endpoint', doc_enc='independent', mlp_size=1000,
                 max_ents=None,
                 emb_size=20, sample_invalid=1.0,
                 label_smoothing_wt=0.0,
                 dataset='litbank', device='cuda', use_gold_ments=False, **kwargs):
        super(BaseController, self).__init__()

        self.device = device
        self.dataset = dataset
        self.use_gold_ments = use_gold_ments
        # Max entities in memory
        self.max_ents = max_ents

        self.max_span_width = max_span_width
        self.top_span_ratio = top_span_ratio
        self.sample_invalid = sample_invalid

        self.label_smoothing_wt = label_smoothing_wt

        if doc_enc == 'independent':
            self.doc_encoder = IndependentDocEncoder(device=self.device, **kwargs)
        else:
            self.doc_encoder = OverlapDocEncoder(device=self.device, **kwargs)

        self.hsize = self.doc_encoder.hsize
        self.mlp_size = mlp_size
        self.emb_size = emb_size
        self.drop_module = nn.Dropout(p=dropout_rate)
        self.ment_emb = ment_emb
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        if self.dataset == 'ontonotes':
            # Ontonotes - Genre embedding
            self.genre_list = ["bc", "bn", "mz", "nw", "pt", "tc", "wb"]
            self.genre_to_idx = dict()
            for idx, genre in enumerate(self.genre_list):
                self.genre_to_idx[genre] = idx

            self.genre_embeddings = nn.Embedding(len(self.genre_list), self.emb_size)

        if self.ment_emb == 'attn':
            self.mention_attn = nn.Linear(self.hsize, 1)

        # Mention modeling part
        self.span_width_embeddings = nn.Embedding(self.max_span_width, self.emb_size)
        self.span_width_prior_embeddings = nn.Embedding(self.max_span_width, self.emb_size)
        self.mention_mlp = MLP(input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
                               hidden_size=self.mlp_size,
                               output_size=1, num_hidden_layers=1, bias=True,
                               drop_module=self.drop_module)
        self.span_width_mlp = MLP(input_size=self.emb_size, hidden_size=self.mlp_size,
                                  output_size=1, num_hidden_layers=1, bias=True,
                                  drop_module=self.drop_module)

        self.memory_net = None
        self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def set_max_ents(self, max_ents):
        self.max_ents = max_ents
        self.memory_net.max_ents = max_ents

    def get_span_embeddings(self, encoded_doc, ment_starts, ment_ends):
        span_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]
        # Add span width embeddings
        span_width_indices = torch.clamp(ment_ends - ment_starts, max=self.max_span_width)
        # print(span_width_indices)
        span_width_embs = self.drop_module(self.span_width_embeddings(span_width_indices))
        span_emb_list.append(span_width_embs)

        if self.ment_emb == 'attn':
            num_words = encoded_doc.shape[0]  # T
            num_c = ment_starts.shape[0]  # C
            doc_range = torch.unsqueeze(torch.arange(num_words, device=self.device), 0).repeat(num_c, 1)  # [C x T]
            ment_masks = ((doc_range >= torch.unsqueeze(ment_starts, dim=1)) &
                          (doc_range <= torch.unsqueeze(ment_ends, dim=1)))  # [C x T]

            word_attn = torch.squeeze(self.mention_attn(encoded_doc), dim=1)  # [T]
            mention_word_attn = nn.functional.softmax(
                (1 - ment_masks.float()) * (-1e10) + torch.unsqueeze(word_attn, dim=0), dim=1)  # [C x T]

            attention_term = torch.matmul(mention_word_attn, encoded_doc)  # K x H
            span_emb_list.append(attention_term)

        return torch.cat(span_emb_list, dim=1)

    def get_mention_width_scores(self, cand_starts, cand_ends):
        span_width_idx = cand_ends - cand_starts
        span_width_embs = self.span_width_prior_embeddings(span_width_idx)
        width_scores = torch.squeeze(self.span_width_mlp(span_width_embs), dim=-1)

        return width_scores

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

    # def get_candidate_endpoints(self, encoded_doc, example):
    #     num_words = encoded_doc.shape[0]
    #
    #     sent_map = torch.tensor(example["sentence_map"], device=self.device)
    #
    #     cand_starts = (torch.unsqueeze(torch.arange(num_words, device=self.device), dim=1)).\
    #         repeat(1, self.max_span_width)
    #     cand_ends = cand_starts + torch.unsqueeze(torch.arange(self.max_span_width, device=self.device), dim=0)
    #
    #     cand_start_sent_indices = sent_map[cand_starts]
    #     # Avoid getting sentence indices for cand_ends >= num_words
    #     corr_cand_ends = torch.min(cand_ends, torch.ones_like(cand_ends, device=self.device) * (num_words - 1))
    #     cand_end_sent_indices = sent_map[corr_cand_ends]
    #
    #     # End before document ends & Same sentence
    #     constraint1 = (cand_ends < num_words)
    #     constraint2 = (cand_start_sent_indices == cand_end_sent_indices)
    #
    #     cand_mask = constraint1 & constraint2
    #     flat_cand_mask = cand_mask.reshape(-1)
    #
    #     # Filter and flatten the candidate end points
    #     filt_cand_starts = cand_starts.reshape(-1)[flat_cand_mask]  # (num_candidates,)
    #     filt_cand_ends = cand_ends.reshape(-1)[flat_cand_mask]  # (num_candidates,)
    #     return filt_cand_starts, filt_cand_ends

    def get_pred_mentions(self, example, encoded_doc, topk=False):
        # num_words = (example["subtoken_map"][-1] - example["subtoken_map"][0] + 1)
        num_words = encoded_doc.shape[0]

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, example)

        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends)

        mention_logits = torch.squeeze(self.mention_mlp(span_embs), dim=-1)
        # Span embeddings not needed anymore
        mention_logits += self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        k = int(self.top_span_ratio * num_words)

        mention_loss = None
        if self.training:
            topk_indices = torch.topk(mention_logits, k)[1]
            filt_gold_mentions = self.get_gold_mentions(example["clusters"], num_words, flat_cand_mask)
            mention_loss = self.mention_loss_fn(mention_logits, filt_gold_mentions)
            # print(topk_indices.shape)
            if not topk:
                # Ignore invalid mentions even during training
                topk_indices = topk_indices[torch.nonzero(filt_gold_mentions[topk_indices], as_tuple=True)[0]]
        else:
            if topk:
                topk_indices = torch.topk(mention_logits, k)[1]
            else:
                topk_indices = torch.squeeze((mention_logits >= 0.0).nonzero(as_tuple=False), dim=1)
                # if k > topk_indices.shape[0]:
                if k < topk_indices.shape[0]:
                    topk_indices = torch.topk(mention_logits, k)[1]

        topk_starts = filt_cand_starts[topk_indices]
        topk_ends = filt_cand_ends[topk_indices]
        topk_scores = mention_logits[topk_indices]

        # Sort the mentions by (start) and tiebreak with (end)
        sort_scores = topk_starts + 1e-5 * topk_ends
        _, sorted_indices = torch.sort(sort_scores, 0)

        return topk_starts[sorted_indices], topk_ends[sorted_indices], topk_scores[sorted_indices], mention_loss

    def get_mention_embs(self, example, topk=False):
        encoded_doc = self.doc_encoder(example)
        mention_loss = None
        if not self.use_gold_ments:
            pred_starts, pred_ends, pred_scores, mention_loss = self.get_pred_mentions(example, encoded_doc, topk=topk)
        else:
            mentions = []
            for cluster in example["clusters"]:
                mentions.extend(cluster)
            pred_starts, pred_ends = zip(*mentions)
            pred_starts = torch.tensor(pred_starts, device=self.device)
            pred_ends = torch.tensor(pred_ends, device=self.device)
            # Fake positive score
            pred_scores = torch.tensor([1.0] * len(mentions), device=self.device)

        # Sort the predicted mentions
        pred_mentions = list(zip(pred_starts.tolist(), pred_ends.tolist()))
        pred_scores = torch.unbind(torch.unsqueeze(pred_scores, dim=1))

        mention_embs = self.get_span_embeddings(encoded_doc, pred_starts, pred_ends)

        mention_emb_list = torch.unbind(mention_embs, dim=0)

        return pred_mentions, mention_emb_list, pred_scores, mention_loss

    def entity_or_not_entity_gt(self, action_tuple_list):
        action_indices = [1 if action_str == 'i' else 0 for (_, action_str) in action_tuple_list]
        action_indices = torch.tensor(action_indices, device=self.device)
        return action_indices

    def calculate_coref_loss(self, action_prob_list, action_tuple_list):
        num_ents = 0
        coref_loss = 0.0
        target_list = []

        # First filter the action tuples to sample invalid
        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            if action_str == 'c':
                gt_idx = cell_idx
            elif action_str == 'o':
                # Overwrite
                gt_idx = (1 if num_ents == 0 else num_ents)
                if self.max_ents is None or num_ents < self.max_ents:
                    num_ents += 1

                if num_ents == 1:
                    # The first ent is always overwritten
                    continue
            else:
                continue

            target = torch.tensor([gt_idx], device=self.device)
            target_list.append(target)

        for idx, target in enumerate(target_list):
            weight = torch.ones_like(action_prob_list[idx], device=self.device)
            if self.training:
                label_smoothing_fn = LabelSmoothingLoss(smoothing=self.label_smoothing_wt, dim=0)
            else:
                label_smoothing_fn = LabelSmoothingLoss(smoothing=0.0, dim=0)

            coref_loss += label_smoothing_fn(pred=action_prob_list[idx], target=target, weight=weight)

        return coref_loss

    def get_genre_embedding(self, examples):
        genre = examples["doc_key"][:2]
        if genre in self.genre_to_idx:
            genre_idx = self.genre_to_idx[genre]
        else:
            genre_idx = self.genre_to_idx['nw']
        return self.genre_embeddings(torch.tensor(genre_idx, device=self.device))

    def forward(self, example, teacher_forcing=False):
        pass
