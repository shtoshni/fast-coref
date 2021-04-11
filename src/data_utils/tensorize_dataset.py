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
        sentences = [([self.tokenizer.cls_token] + sent + [self.tokenizer.sep_token]) for sent in sentences]
        sent_len_list = [len(sent) for sent in sentences]
        max_len = max(sent_len_list)

        if training:
            padded_sent = torch.stack(
                [torch.tensor(self.tokenizer.convert_tokens_to_ids(sent) +
                              (max_len - len(sent)) * [self.tokenizer.pad_token_id], device=self.device)
                 for sent in sentences]
            )
        else:
            # Streaming inference
            padded_sent = [
                torch.unsqueeze(torch.tensor(self.tokenizer.convert_tokens_to_ids(sent), device=self.device), dim=0)
                for sent in sentences]

        output_dict = {"padded_sent": padded_sent,
                       "sentences": sentences,
                       "sent_len_list": sent_len_list,
                       "doc_key": instance["doc_key"],
                       "clusters": instance["clusters"],
                       "subtoken_map": instance["subtoken_map"],
                       "sentence_map": torch.tensor(instance["sentence_map"], device=self.device),
                       }

        return output_dict
