import torch
from os import path
import spacy
from model.utils import action_sequences_to_clusters
from model.entity_ranking_model import EntityRankingModel
from inference.tokenize_doc import tokenize_and_segment_doc, flatten
from omegaconf import OmegaConf


class Inference:
	def __init__(self, model_path, ):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Load model
		checkpoint = torch.load(path.join(model_path, "model.pth"), map_location=self.device)
		self.config = OmegaConf.create(checkpoint['config'])
		self.model = EntityRankingModel(self.config.model, self.config.trainer)
		self._load_model(checkpoint, model_path)

		self.max_segment_len = self.config.model.doc_encoder.transformer.max_segment_len
		self.tokenizer = self.model.mention_proposer.doc_encoder.tokenizer
		self.basic_tokenizer = spacy.load("en_core_web_sm")

	def _load_model(self, checkpoint, model_path):
		self.model.load_state_dict(checkpoint['model'], strict=False)

		if self.config.model.doc_encoder.finetune:
			# Load the document encoder params if encoder is finetuned
			doc_encoder_dir = path.join(model_path, self.config.paths.doc_encoder_dirname)
			# print(doc_encoder_dir)
			# Load the encoder
			from transformers import AutoModel, AutoTokenizer
			self.model.mention_proposer.doc_encoder.lm_encoder = AutoModel.from_pretrained(
				pretrained_model_name_or_path=doc_encoder_dir)
			self.model.mention_proposer.doc_encoder.tokenizer = AutoTokenizer.from_pretrained(
				pretrained_model_name_or_path=doc_encoder_dir)

			# print("Hello")
			if torch.cuda.is_available():
				self.model.cuda()

		self.model.eval()

	@torch.no_grad()
	def perform_coreference(self, document, doc_key="nw"):
		if isinstance(document, str):
			tokenized_doc = tokenize_and_segment_doc(
				document, self.tokenizer, self.basic_tokenizer, max_segment_len=self.max_segment_len)
		elif isinstance(document, dict):
			tokenized_doc = document
		else:
			raise ValueError

		# Ontonotes model need document genre which is formatted as the first two characters of the doc key
		tokenized_doc["doc_key"] = doc_key

		# print(len(tokenized_doc["sentences"]))
		output_doc_dict = tokenized_doc
		doc_tokens = flatten(tokenized_doc["sentences"])

		pred_mentions, _, _, pred_actions = self.model(tokenized_doc)

		idx_clusters = action_sequences_to_clusters(pred_actions, pred_mentions)

		subtoken_map = tokenized_doc["subtoken_map"]
		orig_tokens = tokenized_doc["orig_tokens"]
		clusters = []
		for idx_cluster in idx_clusters:
			cur_cluster = []
			for (ment_start, ment_end) in idx_cluster:
				cur_cluster.append(((ment_start, ment_end),
				                    " ".join(orig_tokens[subtoken_map[ment_start]: subtoken_map[ment_end] + 1])))
				                    # self.tokenizer.convert_ids_to_tokens(doc_tokens[ment_start: ment_end + 1])))

			clusters.append(cur_cluster)

		return {"tokenized_doc": output_doc_dict, "clusters": clusters,
		        "subtoken_idx_clusters": idx_clusters, "actions": pred_actions,
		        "mentions": pred_mentions}


if __name__ == '__main__':
	model_str = "/home/shtoshni/Research/fast-coref/models/ontonotes_best"
	model = Inference(model_str)
	doc = "The practice of referring to Voldemort as \"He Who Must Not Be Named\" might have begun when he used a " \
	      "Taboo. This is, however, unlikely because Dumbledore encouraged using his proper name so as to not fear " \
	      "the name. If saying the Dark Lord’s name would have endangered people, he would not have encouraged it."
	# doc = " Kimberly and Jennifer are friends . The former is a teacher . "
	output_dict = model.perform_coreference(doc)
	print(output_dict["clusters"])
	# print(output_dict)
