"""This is an adaptation of the tokenizer used for LitBank in the overlapping segments setting."""
import torch
from data_processing.overlap_ontonotes import normalize_word


class DocumentState(object):
    def __init__(self):
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.segments = []
        self.real_segments = []
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
            "real_sentences": self.real_segments,
            "sent_len_list": self.sent_len_list,
            "padded_sent": torch.tensor(self.padded_sent),
            "start_indices": self.start_indices,
            "end_indices": self.end_indices,
            'sentence_map': torch.tensor([0] * num_words),  # Assume no sentence boundaries are specified
            "subtoken_map": subtoken_map,
            "part_lens": self.part_lens,
        }


def flatten(l):
  return [item for sublist in l for item in sublist]


def split_into_segments(document_state, constraints1, constraints2, max_segment_len=2048):
    current = 0
    prev_current = -1
    start_idx = 0

    while current < len(document_state.subtokens):
        if prev_current == current:
            break
        # print(current, len(document_state.subtokens))
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

        # print(end)
        if (end + 1) == len(document_state.subtokens):
            end_idx = end + 1
        else:
            last_seg_length = end - current + 1
            # Move current to the middle of last window
            ovlp_current = end - last_seg_length//2
            while ovlp_current < end and not constraints1[ovlp_current]:
                ovlp_current += 1
            # Move to next sentence start token
            ovlp_current += 1
            if ovlp_current == (end + 1):
                ovlp_current = end - last_seg_length//2
                while ovlp_current < end and not constraints2[ovlp_current]:
                    ovlp_current += 1
                # Move to next word
                ovlp_current += 1

            extra_length = (end + 1 - ovlp_current)//2
            end_idx = ovlp_current + extra_length

        document_state.real_segments.append(document_state.subtokens[current:end + 1])
        document_state.segments.append(document_state.subtokens[start_idx:end_idx])
        subtoken_map = document_state.subtoken_map[start_idx: end_idx]
        document_state.segment_subtoken_map.append(subtoken_map)

        document_state.start_indices.append(start_idx - current)
        document_state.end_indices.append(end_idx - current)
        # print(start_idx, end_idx)
        start_idx = end_idx

        if (end + 1) == len(document_state.subtokens):
            current = end + 1
        else:
            current = ovlp_current


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


def tokenize_and_segment_doc(doc_str, lm_tokenizer, max_segment_len=2048):
    document_state = get_tokenized_doc(doc_str, lm_tokenizer)
    document = post_tokenization_processing(document_state, lm_tokenizer, max_segment_len=max_segment_len)
    return document


def tokenize_and_segment_doc_list(doc_list, lm_tokenizer, max_segment_len=2048):
    document_state = DocumentState()
    for doc_str in doc_list:
        get_tokenized_doc(doc_str, lm_tokenizer, document_state=document_state)
        document_state.part_lens.append(len(document_state.tokens))

    document = post_tokenization_processing(document_state, lm_tokenizer, max_segment_len=max_segment_len)
    return document


def post_tokenization_processing(document_state, lm_tokenizer, max_segment_len=2048):
    split_into_segments(document_state, document_state.sentence_end, document_state.token_end,
                        max_segment_len=max_segment_len)

    sentences = [([lm_tokenizer.cls_token] + sent + [lm_tokenizer.sep_token])
                 for sent in document_state.real_segments]
    sent_len_list = [len(sent) for sent in sentences]
    document_state.sent_len_list = sent_len_list
    max_sent_len = max(sent_len_list)
    padded_sent = [lm_tokenizer.convert_tokens_to_ids(sent)
                   + [lm_tokenizer.pad_token_id] * (max_sent_len - len(sent)) for sent in sentences]
    document_state.padded_sent = padded_sent
    return document_state.finalize()


if __name__ == "__main__":
    from transformers import LongformerTokenizerFast

    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096', add_prefix_space=True)
    doc = "My father’s eyes had closed upon the light of this world six months, when Ishmael opened on it."
    print(get_tokenized_doc(doc, tokenizer))

