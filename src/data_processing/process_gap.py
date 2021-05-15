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
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.sentence_map = []
        self.pronouns = []
        self.clusters = []
        self.segment_info = []

    def finalize(self):
        # print(all_mentions)
        num_words = len(flatten(self.segments))
        sentence_map = [0] * num_words
        assert num_words == len(self.subtoken_map), (num_words, len(self.subtoken_map))
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "str_doc": self.tokens,
            "clusters": self.clusters,
            'sentence_map': sentence_map,
            "subtoken_map": self.subtoken_map,
        }


def search_span(word_list, token_list):
    for start_idx in range(0, len(word_list) - len(token_list) + 1):
        match = start_idx
        for token1, token2 in zip(word_list[start_idx: start_idx + len(token_list)], token_list):
            if token1 != token2:
                match = -1
                break

        #         print(word_list, token_list, match)

        if match == -1:
            continue
        else:
            return match

    return -1


def minimize_split(input_dir, output_dir, split="test"):
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096', add_prefix_space=True)

    input_path = path.join(input_dir, "gap-test.tsv")
    output_path = path.join(output_dir, "test.jsonlines".format(split))

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
            pronoun_boundary = (pronoun_offset, pronoun_offset + len(pronoun), 'pronoun')
            span1_boundary = (span1_offset, span1_offset + len(span1), 'A')
            span2_boundary = (span2_offset, span2_offset + len(span2), 'B')

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
            spans = []
            for idx, intermediate_span in enumerate(text_spans):
                prefix_len.append(len(doc_token_list))
                span_tokens = tokenizer.tokenize(intermediate_span)
                if idx % 2 == 1:
                    spans.append([prefix_len[-1], prefix_len[-1] + len(span_tokens) - 1])
                doc_token_list.extend(tokenizer.convert_tokens_to_ids(span_tokens))


            #     #             doc.extend(tokenizer.tokenize(intermediate_span))
            #
            # #
            # # for span_boundary, coref_label in zip([span1_boundary, span2_boundary], [coref1, coref2]):
            # #     boundaries = sorted([pronoun_boundary, span_boundary], key=lambda x: x[0])
            # #
            # #     first_span = text[0: boundaries[0][0]].strip()
            # #     second_span = text[boundaries[0][0]: boundaries[0][1]].strip()
            # #     third_span = text[boundaries[0][1]: boundaries[1][0]].strip()
            # #     fourth_span = text[boundaries[1][0]: boundaries[1][1]].strip()
            # #     fifth_span = text[boundaries[1][1]:].strip()
            #
            #     doc_token_list = []
            #     prefix_len = []
            #     spans = []
            #     for idx, intermediate_span in enumerate([first_span, second_span, third_span, fourth_span, fifth_span]):
            #         prefix_len.append(len(doc_token_list))
            #         span_tokens = tokenizer.tokenize(intermediate_span)
            #         if idx == 1 or idx == 3:
            #             spans.append([prefix_len[-1], prefix_len[-1] + len(span_tokens) - 1])
            #
            #         #             doc.extend(span_tokens)
            #         doc_token_list.extend(tokenizer.convert_tokens_to_ids(span_tokens))
            #     #             doc.extend(tokenizer.tokenize(intermediate_span))
            #
            #     if coref_label == 'TRUE':
            #         clusters = [[spans[0], spans[1]]]
            #     else:
            #         clusters = [[spans[0]], [spans[1]]]

                document = DocumentState(doc_key.strip())
                document.clusters = clusters
                document.tokens = tokenizer.convert_ids_to_tokens(doc_token_list)
                document.segments = [doc_token_list]
                document.subtoken_map = list(range(len(doc_token_list)))

                doc_dict = document.finalize()
                instances_processed += 1
                writer_g.write(json.dumps(doc_dict) + "\n")

    print("Number of instances processed:", instances_processed)
    print(output_path)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    minimize_split(input_dir, output_dir)
