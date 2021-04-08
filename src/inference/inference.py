import torch
import json
from transformers import LongformerTokenizerFast
from auto_memory_model.utils import action_sequences_to_clusters
from auto_memory_model.controller.utils import pick_controller
from inference.tokenize_doc import tokenize_and_segment_doc, tokenize_and_segment_doc_list, flatten


class Inference:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = pick_controller(device=self.device, **checkpoint['model_args'])
        self.max_segment_len = checkpoint['model_args']['max_segment_len']

        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)
        self.model.eval()  # Eval mode

        self.tokenizer = LongformerTokenizerFast.from_pretrained(
            'allenai/longformer-large-4096', add_prefix_space=True)

    def perform_coreference(self, doc, doc_key="nw", num_sents=None):
        if isinstance(doc, str):
            tokenized_doc = tokenize_and_segment_doc(doc, self.tokenizer, max_segment_len=self.max_segment_len)
        elif isinstance(doc, list):
            tokenized_doc = tokenize_and_segment_doc_list(doc, self.tokenizer, max_segment_len=self.max_segment_len)
        elif isinstance(doc, dict):
            tokenized_doc = doc
        else:
            raise ValueError

        # Ontonotes model need document genre which is formatted as the first two characters of the doc key
        tokenized_doc["doc_key"] = doc_key

        # print(len(tokenized_doc["sentences"]))
        output_doc_dict = tokenized_doc
        if num_sents is not None:
            tokenized_doc["sentences"] = tokenized_doc["sentences"][:num_sents]
            output_doc_dict["sentences"] = tokenized_doc["sentences"]
            num_words = sum([len(sentence) for sentence in tokenized_doc["sentences"]])
            output_doc_dict["subtoken_map"] = tokenized_doc["subtoken_map"][:num_words]

        doc_tokens = flatten(tokenized_doc["sentences"])
        subtoken_map = tokenized_doc["subtoken_map"]

        with torch.no_grad():
            pred_actions, pred_mentions, _, _ = self.model(tokenized_doc)

        idx_clusters = action_sequences_to_clusters(pred_actions, pred_mentions)

        mentions = []
        for (ment_start, ment_end) in pred_mentions:
            mentions.append((subtoken_map[ment_start], subtoken_map[ment_end]))

        clusters = []
        for idx_cluster in idx_clusters:
            cur_cluster = []
            for (ment_start, ment_end) in idx_cluster:
                cur_cluster.append(((ment_start, ment_end),
                                    self.tokenizer.convert_tokens_to_string(doc_tokens[ment_start: ment_end + 1])))

            clusters.append(cur_cluster)

        return {"tokenized_doc": output_doc_dict, "clusters": clusters,
                "subtoken_idx_clusters": idx_clusters, "actions": pred_actions,
                "mentions": pred_mentions}
