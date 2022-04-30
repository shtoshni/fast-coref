import torch
from data_processing.utils import split_into_segments, get_sentence_map


class DocumentState:
    def __init__(self):
        self.sentence_end = []
        self.token_end = []
        self.orig_tokens = []
        self.tokens = []
        self.subtokens = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.tensorized_sent = []
        self.sent_len_list = []

    def finalize(self):
        subtoken_map = flatten(self.segment_subtoken_map)
        num_words = len(flatten(self.segments))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))

        return {
            "orig_tokens": self.orig_tokens,
            "sentences": self.segments,
            "sent_len_list": self.sent_len_list,
            "tensorized_sent": self.tensorized_sent,
            "sentence_map": torch.tensor(
                get_sentence_map(self.segments, self.sentence_end)
            ),
            "subtoken_map": subtoken_map,
        }


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_tokenized_doc(doc, subword_tokenizer):
    document_state = DocumentState()

    word_idx = -1
    for sentence in doc:
        for word in sentence:
            document_state.orig_tokens.append(word)
            subtokens = subword_tokenizer.convert_tokens_to_ids(
                subword_tokenizer.tokenize(" " + word)
            )
            document_state.tokens.append(word)
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
            word_idx += 1
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)

        document_state.sentence_end[-1] = True

    # print(document_state.subtokens)
    return document_state


def basic_tokenize_doc(doc_str, basic_tokenizer):
    doc = []
    for sent in basic_tokenizer(doc_str).sents:
        wordlist = [str(word) for word in sent]
        doc.append(wordlist)

    return doc


def tokenize_and_segment_doc(
    basic_tokenized_doc, subword_tokenizer, max_segment_len=4096
):
    document_state: DocumentState = get_tokenized_doc(
        basic_tokenized_doc, subword_tokenizer
    )
    document = post_tokenization_processing(
        document_state, subword_tokenizer, max_segment_len=max_segment_len
    )

    return document


def post_tokenization_processing(
    document_state: DocumentState, subword_tokenizer, max_segment_len=4096
):
    split_into_segments(
        document_state,
        max_segment_len,
        document_state.sentence_end,
        document_state.token_end,
    )

    # sentences = [lm_tokenizer.convert_tokens_to_ids(sent) for sent in document_state.segments]
    sent_len_list = [len(sent) for sent in document_state.segments]
    document_state.sent_len_list = sent_len_list
    document_state.segments_indices = document_state.segments

    # # Tensorize sentence - Streaming coreference is done one window at a time, so no padding is required
    tensorized_sent = [
        torch.unsqueeze(
            torch.tensor(
                [subword_tokenizer.cls_token_id]
                + sent
                + [subword_tokenizer.sep_token_id]
            ),
            dim=0,
        )
        for sent in document_state.segments
    ]
    document_state.tensorized_sent = tensorized_sent
    return document_state.finalize()


if __name__ == "__main__":
    from transformers import LongformerTokenizerFast

    tokenizer = LongformerTokenizerFast.from_pretrained(
        "allenai/longformer-large-4096", add_prefix_space=True
    )
    sample_doc_str = "My father’s eyes had closed upon the light of this world six months, when Ishmael opened on it."
    print(get_tokenized_doc(sample_doc_str, tokenizer))
