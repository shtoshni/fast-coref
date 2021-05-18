from os import path
from collections import Counter, defaultdict

import xml
import xml.etree.ElementTree as ET

import re
import os
import sys
import json

from coref_utils import conll
from os import path
from transformers import LongformerTokenizerFast
from data_processing.utils import flatten

TOTAL_INSTANCES = 273


class DocumentState:
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.segments = []
        self.subtoken_map = []
        self.sentence_map = []
        self.segment_info = []
        self.pronoun_span = []
        self.a_span = []
        self.b_span = []
        self.a_label = 0
        self.b_label = 0

    def finalize(self):
        # print(all_mentions)
        num_words = len(flatten(self.segments))
        sentence_map = [0] * num_words
        assert num_words == len(self.subtoken_map), (num_words, len(self.subtoken_map))
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "str_doc": self.tokens,
            'sentence_map': sentence_map,
            "subtoken_map": self.subtoken_map,
            "pronoun_span": self.pronoun_span,
            "a_span": self.a_span,
            "b_span": self.b_span,
            "a_label": self.a_label,
            "b_label": self.b_label,
        }


def search_span(word_list, token_list):
    for start_idx in range(0, len(word_list) - len(token_list) + 1):
        match = start_idx
        for token1, token2 in zip(word_list[start_idx: start_idx + len(token_list)], token_list):
            if token1 != token2:
                match = -1
                break

        if match == -1:
            continue
        else:
            return match

    return -1


def minimize_partition(input_dir, output_dir, tokenizer, split="test"):
    input_path = path.join(input_dir, f"gap-{split}.tsv")
    output_path = path.join(output_dir, f"{split}.jsonlines")

    instances_processed = 0
    with open(input_path) as reader_f, open(output_path, 'w') as writer_g:
        first_line = True
        for line in reader_f:
            if first_line:
                first_line = False
                continue
            doc_key, text, pronoun, pronoun_offset, span1, span1_offset, coref1,\
                    span2, span2_offset, coref2 = line.strip().split('\t')[:10]

            pronoun_offset, span1_offset, span2_offset = int(pronoun_offset), int(span1_offset), int(span2_offset)
            pronoun_boundary = (pronoun_offset, pronoun_offset + len(pronoun), 'pronoun', None)
            span1_boundary = (span1_offset, span1_offset + len(span1), 'a', coref1 == 'TRUE')
            span2_boundary = (span2_offset, span2_offset + len(span2), 'b', coref2 == 'TRUE')

            span_boundaries = sorted([pronoun_boundary, span1_boundary, span2_boundary], key=lambda x: x[0])
            text_start = 0

            text_spans = []
            for span_boundary in span_boundaries:
                text_spans.append(text[text_start: span_boundary[0]].strip())
                text_spans.append(text[span_boundary[0]: span_boundary[1]].strip())
                text_start = span_boundary[1]

            text_spans.append(text[text_start:].strip())

            doc_token_list = []
            prefix_len = []
            tokenized_spans = []
            for idx, intermediate_span in enumerate(text_spans):
                prefix_len.append(len(doc_token_list))
                span_tokens = tokenizer.tokenize(intermediate_span)
                if idx % 2 == 1:
                    tokenized_spans.append([prefix_len[-1], prefix_len[-1] + len(span_tokens) - 1])
                doc_token_list.extend(tokenizer.convert_tokens_to_ids(span_tokens))

            label_to_span = {}
            label_to_coref_label = {}
            for ((_, _, label, coref_label), tokenized_boundary) in zip(span_boundaries, tokenized_spans):
                label_to_span[label] = tokenized_boundary
                label_to_coref_label[label] = coref_label

            document = DocumentState(doc_key.strip())
            document.tokens = tokenizer.convert_ids_to_tokens(doc_token_list)
            # print(document.tokens)
            # print(text)
            # print(label_to_span)
            # print(len(doc_token_list))
            document.segments = [doc_token_list]
            document.subtoken_map = list(range(len(doc_token_list)))
            document.pronoun_span = label_to_span['pronoun']
            document.a_span = label_to_span['a']
            document.b_span = label_to_span['b']

            document.a_label = label_to_coref_label['a']
            document.b_label = label_to_coref_label['b']

            # print(tokenizer.convert_tokens_to_string(
            #     document.tokens[document.pronoun_span[0]: document.pronoun_span[1] + 1]))
            # print(tokenizer.convert_tokens_to_string(
            #     document.tokens[document.a_span[0]: document.a_span[1] + 1]))
            # print(tokenizer.convert_tokens_to_string(
            #     document.tokens[document.b_span[0]: document.b_span[1] + 1]))

            doc_dict = document.finalize()
            instances_processed += 1
            # break
            writer_g.write(json.dumps(doc_dict) + "\n")

    print("Number of instances processed:", instances_processed)
    print(output_path)


def minimize_split(input_dir, output_dir):
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096', add_prefix_space=True)
    # Create cross validation output dir

    for split in ["validation", "test", "train"]:
        minimize_partition(input_dir, output_dir, tokenizer, split=split)


    # minimize_partition(input_dir, output_dir, split="test")
    # minimize_partition("test", tokenizer, seg_len, input_dir, output_dir)
    # minimize_partition("train", tokenizer, seg_len, input_dir, output_dir)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    minimize_split(input_dir, output_dir)
