import torch
from os import path
from model.utils import action_sequences_to_clusters
from model.entity_ranking_model import EntityRankingModel
from inference.tokenize_doc import tokenize_and_segment_doc, basic_tokenize_doc
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer


class Inference:
    def __init__(self, model_path, encoder_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        checkpoint = torch.load(
            path.join(model_path, "model.pth"), map_location=self.device
        )
        self.config = OmegaConf.create(checkpoint["config"])
        if encoder_name is not None:
            self.config.model.doc_encoder.transformer.model_str = encoder_name
        self.model = EntityRankingModel(self.config.model, self.config.trainer)
        self._load_model(checkpoint, model_path, encoder_name=encoder_name)

        self.max_segment_len = self.config.model.doc_encoder.transformer.max_segment_len
        self.tokenizer = self.model.mention_proposer.doc_encoder.tokenizer

    def _load_model(self, checkpoint, model_path, encoder_name=None):
        self.model.load_state_dict(checkpoint["model"], strict=False)

        if self.config.model.doc_encoder.finetune:
            # Load the document encoder params if encoder is finetuned
            if encoder_name is None:
                doc_encoder_dir = path.join(
                    model_path, self.config.paths.doc_encoder_dirname
                )
                # else:
                # 	doc_encoder_dir = encoder_name
                # Load the encoder
                self.model.mention_proposer.doc_encoder.lm_encoder = (
                    AutoModel.from_pretrained(
                        pretrained_model_name_or_path=doc_encoder_dir
                    )
                )
                self.model.mention_proposer.doc_encoder.tokenizer = (
                    AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=doc_encoder_dir
                    )
                )

            if torch.cuda.is_available():
                self.model.cuda()

        self.model.eval()

    @torch.no_grad()
    def perform_coreference(self, document):
        if isinstance(document, list):
            # Document is already tokenized
            tokenized_doc = tokenize_and_segment_doc(
                document, self.tokenizer, max_segment_len=self.max_segment_len
            )
        elif isinstance(document, str):
            # Raw document string. First perform basic tokenization before further tokenization.
            import spacy

            basic_tokenizer = spacy.load("en_core_web_sm")
            basic_tokenized_doc = basic_tokenize_doc(document, basic_tokenizer)
            tokenized_doc = tokenize_and_segment_doc(
                basic_tokenized_doc,
                self.tokenizer,
                max_segment_len=self.max_segment_len,
            )
        elif isinstance(document, dict):
            tokenized_doc = document
        else:
            raise ValueError

        pred_mentions, _, _, pred_actions = self.model(tokenized_doc)
        idx_clusters = action_sequences_to_clusters(pred_actions, pred_mentions)

        subtoken_map = tokenized_doc["subtoken_map"]
        orig_tokens = tokenized_doc["orig_tokens"]
        clusters = []
        for idx_cluster in idx_clusters:
            cur_cluster = []
            for (ment_start, ment_end) in idx_cluster:
                cur_cluster.append(
                    (
                        (ment_start, ment_end),
                        " ".join(
                            orig_tokens[
                                subtoken_map[ment_start] : subtoken_map[ment_end] + 1
                            ]
                        ),
                    )
                )

            clusters.append(cur_cluster)

        return {
            "tokenized_doc": tokenized_doc,
            "clusters": clusters,
            "subtoken_idx_clusters": idx_clusters,
            "actions": pred_actions,
            "mentions": pred_mentions,
        }


if __name__ == "__main__":
    model_str = "/home/shtoshni/Research/fast-coref/models/ontonotes_best"
    # model = Inference(model_str)
    model = Inference(model_str, "shtoshni/longformer_coreference_ontonotes")

    # doc = " ".join(open("/home/shtoshni/Research/coref_resources/data/ccarol/doc.txt").readlines())
    doc = (
        'The practice of referring to Voldemort as "He Who Must Not Be Named" might have begun when he used a '
        "Taboo. This is, however, unlikely because Dumbledore encouraged using his proper name so as to not fear "
        "the name. If saying the Dark Lord’s name would have endangered people, he would not have encouraged it."
    )
    output_dict = model.perform_coreference(doc)
    print(output_dict["clusters"])
