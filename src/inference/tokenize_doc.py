"""This is an adaptation of the tokenizer used for LitBank in the overlapping segments setting."""
import torch


class DocumentState(object):
    def __init__(self):
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.segments = []
        self.segments_indices = []
        self.start_indices = []
        self.end_indices = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.part_lens = []
        self.padded_sent = []
        self.sent_len_list = []

    def finalize(self):
        subtoken_map = flatten(self.segment_subtoken_map)
        num_words = len(flatten(self.segments))
        # assert num_words == len(flatten(self.speakers))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))

        return {
            "sentences": self.segments,
            "sentences_indices": self.segments_indices,
            "sent_len_list": self.sent_len_list,
            "padded_sent": self.padded_sent,
            "start_indices": self.start_indices,
            "end_indices": self.end_indices,
            'sentence_map': torch.tensor([0] * num_words),  # Assume no sentence boundaries are specified
            "subtoken_map": subtoken_map,
            "part_lens": self.part_lens,
        }


def flatten(l):
    return [item for sublist in l for item in sublist]


def split_into_segments_independent(document_state, constraints1, constraints2, max_segment_len=4096):
    current = 0
    # previous_token = 0
    while current < len(document_state.subtokens):
        end = min(current + max_segment_len - 1 - 2,
                  len(document_state.subtokens) - 1)
        while end >= current and not constraints1[end]:
            end -= 1
        if end < current:
            end = min(current + max_segment_len - 1 - 2,
                      len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")
        document_state.segments.append(document_state.subtokens[current:end + 1])
        subtoken_map = document_state.subtoken_map[current: end + 1]
        document_state.segment_subtoken_map.append(subtoken_map)
        current = end + 1


def get_tokenized_doc(doc_str, tokenizer, document_state=None):
    word_idx = -1

    if document_state is None:
        document_state = DocumentState()
    else:
        if len(document_state.subtoken_map):
            word_idx = document_state.subtoken_map[-1]

    tokenized_doc = tokenizer.tokenize(doc_str)

    for idx, token in enumerate(tokenized_doc):
        word_idx += 1
        document_state.tokens.append(token)
        # Subtoken and token are same
        document_state.subtokens.append(token)
        if idx == len(tokenized_doc) - 1:
            # End of document
            document_state.token_end += ([True])
        else:
            document_state.token_end += ([True])

        document_state.subtoken_map.append(word_idx)
        document_state.sentence_end.append(False)  # No info on sentence end
    return document_state


def tokenize_and_segment_doc(doc_str, lm_tokenizer, max_segment_len=4096):
    document_state = get_tokenized_doc(doc_str, lm_tokenizer)
    document = post_tokenization_processing(
        document_state, lm_tokenizer, max_segment_len=max_segment_len)
    return document


def tokenize_and_segment_doc_list(doc_list, lm_tokenizer, max_segment_len=4096):
    document_state = DocumentState()
    for doc_str in doc_list:
        get_tokenized_doc(doc_str, lm_tokenizer, document_state=document_state)
        document_state.part_lens.append(len(document_state.tokens))

    document = post_tokenization_processing(document_state, lm_tokenizer, max_segment_len=max_segment_len)
    return document


def post_tokenization_processing(document_state, lm_tokenizer, max_segment_len=4096):
    split_into_segments_independent(
        document_state, document_state.sentence_end, document_state.token_end, max_segment_len=max_segment_len)

    sentences = [lm_tokenizer.convert_tokens_to_ids(sent) for sent in document_state.segments]
    sent_len_list = [len(sent) for sent in sentences]
    document_state.sent_len_list = sent_len_list
    document_state.segments_indices = sentences

    # Tensorize sentence - Streaming is done one window at a time, so no padding required
    padded_sent = [torch.unsqueeze(
        torch.tensor([lm_tokenizer.cls_token_id] + sent + [lm_tokenizer.sep_token_id]), dim=0) for sent in sentences]
    document_state.padded_sent = padded_sent
    return document_state.finalize()


if __name__ == "__main__":
    from transformers import LongformerTokenizerFast

    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096', add_prefix_space=True)
    doc = "My father’s eyes had closed upon the light of this world six months, when Ishmael opened on it."
    print(get_tokenized_doc(doc, tokenizer))

