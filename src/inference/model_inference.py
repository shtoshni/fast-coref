import torch
from transformers import LongformerTokenizerFast
from memory_model.utils import action_sequences_to_clusters
from memory_model.controller.utils import pick_controller
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

    def perform_coreference(self, document, doc_key="nw"):
        if isinstance(document, str):
            tokenized_doc = tokenize_and_segment_doc(
                document, self.tokenizer, max_segment_len=self.max_segment_len)
        elif isinstance(document, list):
            tokenized_doc = tokenize_and_segment_doc_list(
                document, self.tokenizer, max_segment_len=self.max_segment_len)
        elif isinstance(document, dict):
            tokenized_doc = document
        else:
            raise ValueError

        # Ontonotes model need document genre which is formatted as the first two characters of the doc key
        tokenized_doc["doc_key"] = doc_key

        # print(len(tokenized_doc["sentences"]))
        output_doc_dict = tokenized_doc
        doc_tokens = flatten(tokenized_doc["sentences"])

        with torch.no_grad():
            pred_actions, pred_mentions = self.model(tokenized_doc)[:2]

        idx_clusters = action_sequences_to_clusters(pred_actions, pred_mentions)

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


if __name__ == '__main__':
    doc = open("/home/shtoshni/Research/litbank_coref/data/doc.txt").readlines()
    print(doc)

    model_path = "/home/shtoshni/Research/fast-coref/models/joint_downsample_500/model.pth"
    model = Inference(model_path)
    model.model.max_span_width = 10
    # doc = "The practice of referring to Voldemort as \"He Who Must Not Be Named\" might have begun when he used a " \
    #       "Taboo. This is, however, unlikely because Dumbledore encouraged using his proper name so as to not fear " \
    #       "the name. If saying the Dark Lord’s name would have endangered people, he would not have encouraged it."
    # doc = " Kimberly and Jennifer are friends . The former is a teacher . "
    output_dict = model.perform_coreference(doc)
    print(output_dict["clusters"])
