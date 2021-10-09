import torch
import torch.nn as nn

from model.document_encoder import IndependentDocEncoder
from pytorch_utils.modules import MLP

from model.mention_proposal.utils import transform_gold_mentions, sort_mentions


class MentionProposalModule(nn.Module):
	def __init__(self, model_config, drop_module=None):
		super(MentionProposalModule, self).__init__()

		self.model_config = model_config
		self.drop_module = drop_module

		# Encoder
		self.doc_encoder = IndependentDocEncoder(model_config.doc_encoder)

		# Mention proposal model
		self.mention_params = self.model_config.mention_params
		self._build_model(
			mention_params=self.mention_params,
			hidden_size=self.doc_encoder.hidden_size)

		self.loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

	@property
	def device(self) -> torch.device:
		""" A workaround to get current device (which is assumed to be the
		device of the first parameter of one of the submodules) """
		return next(self.doc_encoder.parameters()).device

	def _build_model(self, mention_params, hidden_size):
		self.span_width_embeddings = nn.Embedding(
			mention_params.max_span_width, mention_params.emb_size)
		self.span_width_prior_embeddings = nn.Embedding(
			mention_params.max_span_width, mention_params.emb_size)

		ment_emb_type = self.model_config.mention_params.ment_emb
		ment_emb_to_size_factor = mention_params.ment_emb_to_size_factor[ment_emb_type]

		if ment_emb_type == 'attn':
			self.mention_attn = nn.Linear(hidden_size, 1).to(self.device)

		self.span_emb_size = ment_emb_to_size_factor * hidden_size + mention_params.emb_size
		self.mention_mlp = MLP(
			input_size=self.span_emb_size, hidden_size=mention_params.mlp_size, output_size=1,
			bias=True, drop_module=self.drop_module, num_hidden_layers=mention_params.mlp_depth
		)
		self.span_width_mlp = MLP(
			input_size=mention_params.emb_size, hidden_size=mention_params.mlp_size,
			output_size=1, num_hidden_layers=mention_params.mlp_depth, bias=True,
			drop_module=self.drop_module
		)

	def get_params(self, named=False):
		encoder_params, mem_params = [], []
		for name, param in self.named_parameters():
			elem = (name, param) if named else param
			if name.startswith('doc_encoder'):
				encoder_params.append(elem)
			else:
				mem_params.append(elem)

		return encoder_params, mem_params

	def get_span_embeddings(self, encoded_doc, ment_starts, ment_ends):
		span_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]
		# Add span width embeddings
		span_width_indices = torch.clamp(ment_ends - ment_starts, max=self.mention_params.max_span_width - 1)
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
		span_width_idx = torch.clamp(cand_ends - cand_starts, max=self.mention_params.max_span_width - 1)
		span_width_embs = self.span_width_prior_embeddings(span_width_idx)
		width_scores = torch.squeeze(self.span_width_mlp(span_width_embs), dim=-1)

		return width_scores

	def get_flat_gold_mentions(self, clusters, num_tokens, flat_cand_mask):
		gold_ments = torch.zeros(num_tokens, self.mentions_params.max_span_width, device=self.device)
		for cluster in clusters:
			for mention in cluster:
				span_start, span_end = mention[:2]
				span_width = span_end - span_start + 1
				if span_width <= self.mention_params.max_span_width:
					span_width_idx = span_width - 1
					gold_ments[span_start, span_width_idx] = 1

		filt_gold_ments = gold_ments.reshape(-1)[flat_cand_mask].float()
		# assert(torch.sum(gold_ments) == torch.sum(filt_gold_ments))  # Filtering shouldn't remove gold mentions
		return filt_gold_ments

	def get_candidate_endpoints(self, encoded_doc, document):
		num_words = encoded_doc.shape[0]
		sent_map = document["sentence_map"].to(self.device)
		# num_words x max_span_width
		cand_starts = torch.unsqueeze(torch.arange(num_words, device=self.device), dim=1).repeat(1, self.max_span_width)
		cand_ends = cand_starts + torch.unsqueeze(
			torch.arange(self.mention_params.max_span_width, device=self.device), dim=0)

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

	def get_pred_mentions(self, document, encoded_doc, ment_threshold=0.0) -> dict:
		num_tokens = encoded_doc.shape[0]
		num_words = document["subtoken_map"][-1] - document["subtoken_map"][0] + 1
		cand_starts, cand_ends, cand_mask = self.get_candidate_endpoints(encoded_doc, document)

		span_embs = self.get_span_embeddings(encoded_doc, cand_starts, cand_ends)

		mention_logits = torch.squeeze(self.mention_mlp(span_embs), dim=-1)
		mention_logits += self.get_mention_width_scores(cand_starts, cand_ends)

		del span_embs  # Span embeddings not required anymore

		output_dict = {}
		if self.training:
			k = int(self.mention_params.top_span_ratio * num_words)
			topk_indices = torch.topk(mention_logits, k)[1]
			filt_gold_mentions = self.get_flat_gold_mentions(document["clusters"], num_tokens, cand_mask)

			if self.mention_params.ment_loss == 'all':
				mention_loss = self.loss_fn(mention_logits, filt_gold_mentions)
			else:
				mention_loss = self.loss_fn(mention_logits[topk_indices], filt_gold_mentions[topk_indices])

			# Add mention loss to output
			output_dict['mention_loss'] = mention_loss

			if not self.mention_params.use_topk:
				# Ignore invalid mentions even during training
				topk_indices = topk_indices[torch.nonzero(filt_gold_mentions[topk_indices], as_tuple=True)[0]]
		else:
			if self.mention_params.use_topk:
				k = int(self.mention_params.top_span_ratio * num_words)
				topk_indices = torch.topk(mention_logits, k)[1]
			else:
				topk_indices = torch.squeeze((mention_logits >= ment_threshold).nonzero(as_tuple=False), dim=1)

		topk_starts = cand_starts[topk_indices]
		topk_ends = cand_ends[topk_indices]
		topk_scores = mention_logits[topk_indices]

		output_dict['ment_starts'], output_dict['ment_ends'], sorted_indices = \
			sort_mentions(topk_starts, topk_ends, return_sorted_indices=True)

		output_dict['ment_scores'] = topk_scores[sorted_indices]

		return output_dict

	def forward(self, document):
		"Given the docume mention embeddings for the proposed mentions."
		encoded_doc = self.doc_encoder(document)

		if self.mention_params.use_gold_ments:
			# Process gold mentions to a format similar to mentions obtained after prediction
			output_dict = transform_gold_mentions(document)
		else:
			output_dict = self.get_pred_mentions(document, encoded_doc)

		pred_starts, pred_ends = output_dict['ment_starts'], output_dict['ment_ends']

		# Stack the starts and ends to get the mention tuple
		output_dict['ments'] = torch.stack((pred_starts, pred_ends), dim=1)
		# Get mention embeddings
		mention_embs = self.get_span_embeddings(encoded_doc, pred_starts, pred_ends)
		output_dict['ment_emb_list'] = torch.unbind(mention_embs, dim=0)

		return output_dict
