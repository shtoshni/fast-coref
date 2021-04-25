import torch

from transformers import LongformerTokenizerFast


class TensorizeDataset:
    def __init__(self, doc_enc, device=None):
        self.doc_enc = doc_enc
        self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096')
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def tensorize_data(self, split_data, training=False):
        tensorized_data = []
        for instance in split_data:
            tensorized_data.append(self.tensorize_instance_independent(
                instance, training=training))

        return tensorized_data

    def tensorize_instance_independent(self, instance, training=False):
        sentences = instance["sentences"]
        clusters = instance["clusters"]
        sentence_map = instance["sentence_map"]
        subtoken_map = instance["subtoken_map"]

        sentences = [([self.tokenizer.cls_token] + sent + [self.tokenizer.sep_token]) for sent in sentences]

        if training:
            if len(sentences) > 1:
                # Truncate to prefix - happens rarely for segment length of 4096
                sentences = sentences[:1]
                num_words = len(sentences[0]) - 2  # Remove special tokens
                sentence_map = sentence_map[:num_words]
                subtoken_map = subtoken_map[:num_words]
                clusters = []
                for orig_cluster in instance["clusters"]:
                    cluster = []
                    for ment_start, ment_end in orig_cluster:
                        if ment_end < num_words:
                            cluster.append((ment_start, ment_end))

                    if len(cluster):
                        clusters.append(cluster)

            padded_sent = torch.unsqueeze(
                torch.tensor(self.tokenizer.convert_tokens_to_ids(sentences[0]), device=self.device), dim=0)
            sent_len_list = [len(sent) for sent in sentences]
        else:
            # Streaming inference
            sent_len_list = [len(sent) for sent in sentences]
            padded_sent = [
                torch.unsqueeze(torch.tensor(self.tokenizer.convert_tokens_to_ids(sent), device=self.device), dim=0)
                for sent in sentences]

        output_dict = {"padded_sent": padded_sent,
                       "sentences": sentences,
                       "sent_len_list": sent_len_list,
                       "doc_key": instance["doc_key"],
                       "clusters": clusters,
                       "subtoken_map": subtoken_map,
                       "sentence_map": torch.tensor(sentence_map, device=self.device),
                       }

        return output_dict
