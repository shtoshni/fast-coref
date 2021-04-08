from copy import deepcopy
import random
import torch

from transformers import LongformerTokenizerFast


class TensorizeDataset:
    def __init__(self, device=None):
        self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096')
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def tensorize_data(self, split_data, max_training_segments=None):
        tensorized_data = []
        for instance in split_data:
            tensorized_data.append(self.tensorize_instance(instance, max_training_segments=max_training_segments))

        return tensorized_data

    def tensorize_instance(self, instance, max_training_segments=None):
        num_sentences = len(instance["real_sentences"])
        if max_training_segments is not None and num_sentences > max_training_segments:
            instance = deepcopy(instance)
            sentence_offset = random.randint(0, num_sentences - max_training_segments)
            word_offset = sum([(end_idx - start_idx)
                               for start_idx, end_idx in zip(instance["start_indices"][:sentence_offset],
                                                             instance["end_indices"][:sentence_offset])])
            sentences = instance["real_sentences"][sentence_offset: sentence_offset + max_training_segments]

            start_indices = instance["start_indices"][sentence_offset: sentence_offset + max_training_segments]
            # Set first window to start at 0th token
            word_offset -= start_indices[0]
            start_indices[0] = 0

            end_indices = instance["end_indices"][sentence_offset: sentence_offset + max_training_segments]
            # Set last window to end at last token
            end_indices[-1] = len(sentences[-1])

            num_words = sum([(end_idx - start_idx) for start_idx, end_idx in zip(start_indices, end_indices)])
            sentence_map = instance["sentence_map"][word_offset: word_offset + num_words]

            clusters = []
            for orig_cluster in instance["clusters"]:
                cluster = []
                for ment_start, ment_end in orig_cluster:
                    if ment_end >= word_offset and ment_start < word_offset + num_words:
                        cluster.append((ment_start - word_offset, ment_end - word_offset))

                if len(cluster):
                    clusters.append(cluster)

            instance["real_sentences"] = sentences
            instance["clusters"] = clusters
            instance["sentence_map"] = sentence_map
            instance["start_indices"] = start_indices
            instance["end_indices"] = end_indices

        output_dict = {}
        sentences = [([self.tokenizer.cls_token] + sent + [self.tokenizer.sep_token])
                     for sent in instance["real_sentences"]]
        sent_len_list = [len(sent) for sent in sentences]
        max_sent_len = max(sent_len_list)
        padded_sent = [self.tokenizer.convert_tokens_to_ids(sent)
                       + [self.tokenizer.pad_token_id] * (max_sent_len - len(sent)) for sent in sentences]

        output_dict["padded_sent"] = torch.tensor(padded_sent, device=self.device)
        output_dict["sent_len_list"] = torch.tensor(sent_len_list, device=self.device)

        output_dict["doc_key"] = instance["doc_key"]
        output_dict["clusters"] = instance["clusters"]
        output_dict["subtoken_map"] = instance["subtoken_map"]
        output_dict["sentence_map"] = torch.tensor(instance["sentence_map"], device=self.device)
        output_dict["start_indices"] = instance["start_indices"]
        output_dict["end_indices"] = instance["end_indices"]

        return output_dict
