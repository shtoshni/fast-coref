import torch

from transformers import LongformerTokenizerFast


class TensorizeDataset:
    def __init__(self, remove_singletons=False):
        self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096')
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.remove_singletons = remove_singletons

    def tensorize_data(self, split_data, training=False):
        tensorized_data = []
        for instance in split_data:
            tensorized_data.append(self.tensorize_instance_independent(
                instance, training=training))

        return tensorized_data

    def process_sentence(self, sentence):
        return [self.tokenizer.cls_token_id] + sentence + [self.tokenizer.sep_token_id]

    def tensorize_instance_independent(self, instance, training=False):
        sentences = instance["sentences"]
        clusters = instance["clusters"]
        sentence_map = instance["sentence_map"]
        subtoken_map = instance["subtoken_map"]

        if training:
            if len(sentences) > 1:
                # Truncate to prefix - happens rarely for segment length of 4096
                # For segments <= 2048 it does happen
                sentences = sentences[:1]
                num_words = len(sentences[0])
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

            tensorized_sent = torch.unsqueeze(
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
                       "doc_key": instance["doc_key"],
                       "clusters": clusters,
                       "subtoken_map": subtoken_map,
                       "sentence_map": torch.tensor(sentence_map, device=self.device),
                       }

        if self.remove_singletons:
            output_dict['clusters'] = [cluster for cluster in output_dict['clusters'] if len(cluster) > 1]

        return output_dict
