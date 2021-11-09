import torch
from typing import List, Dict, Union
from transformers import PreTrainedTokenizerFast
from torch import Tensor


class TensorizeDataset:
	def __init__(self, tokenizer: PreTrainedTokenizerFast, remove_singletons: bool = False) -> None:
		self.tokenizer = tokenizer
		self.remove_singletons = remove_singletons
		self.device = torch.device("cpu")

	def tensorize_data(self, split_data: List[Dict], training: bool = False) -> List[Dict]:
		tensorized_data = []
		for document in split_data:
			tensorized_data.append(
				self.tensorize_instance_independent(document, training=training))

		return tensorized_data

	def process_sentence(self, sentence: List) -> List:
		return [self.tokenizer.cls_token_id] + sentence + [self.tokenizer.sep_token_id]

	def tensorize_instance_independent(self, document: Dict, training: bool = False) -> Dict:
		sentences: List[List[int]] = document["sentences"]
		clusters: List = document.get("clusters", [])
		sentence_map: List[int] = document["sentence_map"]
		subtoken_map: List[int] = document["subtoken_map"]

		if training:
			if len(sentences) > 1:
				# Truncate to prefix - happens rarely for segment length of 4096
				# For segments <= 2048 it does happen reasonably frequently for LitBank and
				# a few documents in OntoNotes.
				sentences = sentences[:1]
				num_words: int = len(sentences[0])
				sentence_map = sentence_map[:num_words]
				subtoken_map = subtoken_map[:num_words]
				clusters = []
				for orig_cluster in document["clusters"]:
					cluster = []
					for ment_start, ment_end in orig_cluster:
						if ment_end < num_words:
							cluster.append((ment_start, ment_end))

					if len(cluster):
						clusters.append(cluster)

			tensorized_sent: Union[List[Tensor], Tensor] = torch.unsqueeze(
				torch.tensor(self.process_sentence(sentences[0]), device=self.device), dim=0)

		else:
			# Streaming inference
			tensorized_sent = [
				torch.unsqueeze(torch.tensor(self.process_sentence(sent), device=self.device), dim=0)
				for sent in sentences]

		sent_len_list = [len(sent) for sent in sentences]
		output_dict = {"tensorized_sent": tensorized_sent,
		               "sentences": sentences,
		               "sent_len_list": sent_len_list,
		               "doc_key": document.get("doc_key", None),
		               "clusters": clusters,
		               "subtoken_map": subtoken_map,
		               "sentence_map": torch.tensor(sentence_map, device=self.device),
		               }

		# Pass along other metadata
		for key in document:
			if key not in output_dict:
				output_dict[key] = document[key]

		if self.remove_singletons:
			output_dict['clusters'] = [cluster for cluster in output_dict['clusters'] if len(cluster) > 1]

		return output_dict
