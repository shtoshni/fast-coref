import torch
import torch.nn as nn

from document_encoder.independent import IndependentDocEncoder
from pytorch_utils.modules import MLP
from pytorch_utils.label_smoothing import LabelSmoothingLoss


class BaseController(nn.Module):
    def __init__(self, dropout_rate=0.5, **kwargs):
        super(BaseController, self).__init__()
        self.__dict__.update(kwargs)

        print("Basecontroller:", self.device)
        if torch.cuda.device_count() > 1:
            kwargs['device'] = torch.device('cuda:1')

        self.doc_encoder = IndependentDocEncoder(**kwargs)
        self.hsize = self.doc_encoder.hsize
        self.drop_module = nn.Dropout(p=dropout_rate)
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        if self.ment_emb == 'attn':
            self.mention_attn = nn.Linear(self.hsize, 1).to(self.device)

        self.num_feats = 2
        self.doc_class_to_idx = {}
        if self.doc_class is not None:
            self.num_feats = 3
            # Ontonotes - Genre embedding
            if self.doc_class == 'dialog':
                self.doc_class_to_idx = {'bc': 0, 'tc': 0, 'bn': 1, 'mz': 1, 'nw': 1, 'pt': 1, 'wb': 1}
                self.genre_embeddings = nn.Embedding(2, self.emb_size).to(self.device)
            else:
                genre_list = ["bc", "bn", "mz", "nw", "pt", "tc", "wb"]
                for idx, genre in enumerate(genre_list):
                    self.doc_class_to_idx[genre] = idx
                self.genre_embeddings = nn.Embedding(len(genre_list), self.emb_size).to(self.device)

        # Mention modeling part
        self.span_width_embeddings = nn.Embedding(self.max_span_width, self.emb_size).to(self.device)
        self.span_width_prior_embeddings = nn.Embedding(self.max_span_width, self.emb_size).to(self.device)
        self.mention_mlp = MLP(input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
                               hidden_size=self.mlp_size,
                               output_size=1, num_hidden_layers=1, bias=True,
                               drop_module=self.drop_module).to(self.device)
        self.span_width_mlp = MLP(input_size=self.emb_size, hidden_size=self.mlp_size,
                                  output_size=1, num_hidden_layers=1, bias=True,
                                  drop_module=self.drop_module).to(self.device)

        self.memory_net = None
        self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        if self.normalize_loss:
            self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)

    def get_tokenizer(self):
        return self.doc_encoder.get_tokenizer()

    def to_add_speaker_tokens(self):
        return self.doc_encoder.to_add_speaker_tokens()

    def get_params(self, named=False):
        encoder_params, mem_params = [], []
        for name, param in self.named_parameters():
            elem = (name, param) if named else param
            if name.startswith('doc_encoder'):
                encoder_params.append(elem)
            else:
                mem_params.append(elem)

        return encoder_params, mem_params

    def set_max_ents(self, max_ents):
        self.max_ents = max_ents
        self.memory_net.max_ents = max_ents

    def get_span_embeddings(self, encoded_doc, ment_starts, ment_ends):
        span_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]
        # Add span width embeddings
        span_width_indices = torch.clamp(ment_ends - ment_starts, max=self.max_span_width - 1)
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
        span_width_idx = torch.clamp(cand_ends - cand_starts, max=self.max_span_width - 1)
        span_width_embs = self.span_width_prior_embeddings(span_width_idx)
        width_scores = torch.squeeze(self.span_width_mlp(span_width_embs), dim=-1)

        return width_scores

    def get_gold_mentions(self, clusters, num_words, flat_cand_mask):
        gold_ments = torch.zeros(num_words, self.max_span_width, device=self.device)
        for cluster_idx, cluster in enumerate(clusters):
            for mention in cluster:
                span_start, span_end = mention[:2]
                span_width = span_end - span_start + 1
                if span_width <= self.max_span_width:
                    span_width_idx = span_width - 1
                    gold_ments[span_start, span_width_idx] = cluster_idx + 1

        filt_gold_ments = gold_ments.reshape(-1)[flat_cand_mask].float()
        # assert(torch.sum(gold_ments) == torch.sum(filt_gold_ments))  # Filtering shouldn't remove gold mentions
        return filt_gold_ments

    def get_candidate_endpoints(self, encoded_doc, instance):
        num_words = encoded_doc.shape[0]
        sent_map = instance["sentence_map"].to(self.device)
        # num_words x max_span_width
        cand_starts = torch.unsqueeze(torch.arange(num_words, device=self.device), dim=1).repeat(1, self.max_span_width)
        cand_ends = cand_starts + torch.unsqueeze(torch.arange(self.max_span_width, device=self.device), dim=0)

        cand_start_sent_indices = sent_map[cand_starts]
        # Avoid getting sentence indices for cand_ends >= num_words
        corr_cand_ends = torch.min(cand_ends, torch.ones_like(cand_ends, device=self.device) * (num_words - 1))
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

    def get_pred_mentions(self, instance, encoded_doc):
        num_words = encoded_doc.shape[0]

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, instance)

        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends)

        mention_logits = torch.squeeze(self.mention_mlp(span_embs), dim=-1)
        # Span embeddings not needed anymore
        mention_logits += self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        train_vars = {}
        if self.training:
            k = int(self.top_span_ratio * (instance["subtoken_map"][-1] - instance["subtoken_map"][0] + 1))
            topk_indices = torch.topk(mention_logits, k)[1]
            filt_gold_mentions = self.get_gold_mentions(instance["clusters"], num_words, flat_cand_mask)
            # mention_loss = self.mention_loss_fn(mention_logits, filt_gold_mentions)
            if self.ment_loss == 'all':
                mention_loss = self.mention_loss_fn(mention_logits, torch.clamp(filt_gold_mentions, 0, 1))
            else:
                mention_loss = self.mention_loss_fn(mention_logits[topk_indices],
                                                    torch.clamp(filt_gold_mentions[topk_indices], 0, 1))

            uniq_cluster_count = torch.unique(filt_gold_mentions[topk_indices]).shape[0]

            train_vars["mention_loss"] = mention_loss
            train_vars["uniq_cluster_count"] = uniq_cluster_count
            if not self.use_topk:
                # Ignore invalid mentions even during training
                topk_indices = topk_indices[torch.nonzero(filt_gold_mentions[topk_indices], as_tuple=True)[0]]
        else:
            if self.use_topk:
                k = int(self.top_span_ratio * (instance["subtoken_map"][-1] - instance["subtoken_map"][0] + 1))
                topk_indices = torch.topk(mention_logits, k)[1]
            else:
                topk_indices = torch.squeeze((mention_logits >= 0.0).nonzero(as_tuple=False), dim=1)

        topk_starts = filt_cand_starts[topk_indices]
        topk_ends = filt_cand_ends[topk_indices]
        topk_scores = mention_logits[topk_indices]

        # Sort the mentions by (start) and tiebreak with (end)
        sort_scores = topk_starts + 1e-5 * topk_ends
        _, sorted_indices = torch.sort(sort_scores, 0)

        return topk_starts[sorted_indices], topk_ends[sorted_indices], topk_scores[sorted_indices], train_vars

    def get_mention_embs(self, instance):
        encoded_doc = self.doc_encoder(instance)
        if torch.cuda.device_count() > 1:
            encoded_doc = encoded_doc.to(self.device)
        train_vars = None
        if not self.use_gold_ments:
            pred_starts, pred_ends, pred_scores, train_vars = self.get_pred_mentions(instance, encoded_doc)
        else:
            mentions = []
            for cluster in instance["clusters"]:
                for ment_start, ment_end in cluster:
                    mentions.append((ment_start, ment_end))

            if len(mentions):
                topk_starts, topk_ends = zip(*mentions)
            else:
                return None, [], None, {}

            topk_starts = torch.tensor(topk_starts, device=self.device)
            topk_ends = torch.tensor(topk_ends, device=self.device)
            # Fake positive score
            pred_scores = torch.tensor([1.0] * len(mentions), device=self.device)

            sort_scores = topk_starts + 1e-5 * topk_ends
            _, sorted_indices = torch.sort(sort_scores, 0)

            pred_starts, pred_ends = topk_starts[sorted_indices], topk_ends[sorted_indices]

        # Sort the predicted mentions
        pred_mentions = torch.stack((pred_starts, pred_ends), dim=1)

        mention_embs = self.get_span_embeddings(encoded_doc, pred_starts, pred_ends)
        mention_emb_list = torch.unbind(mention_embs, dim=0)

        return pred_mentions, mention_emb_list, pred_scores, train_vars

    def calculate_coref_loss(self, action_prob_list, action_tuple_list):
        num_ents = 0
        coref_loss = 0.0
        counter = 0

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
            weight = torch.ones_like(action_prob_list[counter], device=self.device)
            label_smoothing_fn = LabelSmoothingLoss(smoothing=self.label_smoothing_wt, dim=0)

            coref_loss += label_smoothing_fn(pred=action_prob_list[counter], target=target, weight=weight)
            counter += 1

        return coref_loss

    def get_metadata(self, instance):
        if self.doc_class_to_idx:
            doc_class = instance["doc_key"][:2]
            if doc_class in self.doc_class_to_idx:
                doc_class_idx = self.doc_class_to_idx[doc_class]
            else:
                doc_class_idx = self.doc_class_to_idx['nw']  # Non-dialog
            return {'genre': self.genre_embeddings(torch.tensor(doc_class_idx, device=self.device))}
        else:
            return {}

    def forward(self, instance, teacher_forcing=False):
        pass
