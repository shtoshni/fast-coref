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
        all_mentions = flatten(self.clusters)
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

    input_path = path.join(input_dir, "WSCollection.xml")
    output_path = path.join(output_dir, "test.jsonlines".format(split))
    not_found_count = 0
    instances_processed = 0

    tree = ET.parse(input_path)
    root = tree.getroot()

    prefixes = []
    pronouns = []
    continuations = []

    answers = []
    correct_answers = []

    with open(output_path, 'w') as out_f:
        for elem in list(root)[:TOTAL_INSTANCES]:
            for children in list(elem.iter('txt1')):
                prefixes.append(children.text.strip().replace('\n', ' '))

            for children in list(elem.iter('pron')):
                pronouns.append(children.text.strip())

            for children in list(elem.iter('txt2')):
                continuations.append(children.text.strip())

            for children in list(elem.iter('answer')):
                answers.append(children.text.strip())

            for children in list(elem.iter('correctAnswer')):
                correct_answers.append(children.text.strip()[0])

        for idx, prefix in enumerate(prefixes):
            answer1 = answers[idx * 2]
            answer2 = answers[idx * 2 + 1]

            text = f'{prefix} {pronouns[idx * 2]} {continuations[idx]}'
            word_list = tokenizer.tokenize(prefix)
            prefix_idx = len(word_list)
            word_list += tokenizer.tokenize(pronouns[idx * 2])

            pronoun_boundary = [prefix_idx, len(word_list) - 1]
            word_list += tokenizer.tokenize(continuations[idx])

            answer_boundaries = []

            for answer in [answer1, answer2]:
                for span in [answer, answer.lower(), answer.capitalize()]:
                    span_tokens = tokenizer.tokenize(span)
                    found = search_span(word_list, span_tokens)
                    if found != -1:
                        answer_boundaries.append([[found, found + len(span_tokens) - 1]])
                        break

                if found == -1:
                    print(text, answer)
                    not_found_count += 1

            import copy
            clusters = copy.deepcopy(answer_boundaries)
            if len(answer_boundaries) == 2:
                correct_answer = correct_answers[idx]
                assert (correct_answer in ['A', 'B'])

                if correct_answer == 'A':
                    cluster_idx = 0
                else:
                    cluster_idx = 1

                clusters[cluster_idx].append(pronoun_boundary)
                clusters[cluster_idx] = sorted(clusters[cluster_idx], key=lambda x: x[0])

                document = DocumentState(f'wsc_{idx}')
                document.clusters = clusters
                document.tokens = word_list
                document.segments = [tokenizer.convert_tokens_to_ids(word_list)]
                document.subtoken_map = list(range(len(word_list)))

                doc_dict = document.finalize()
                instances_processed += 1
                out_f.write(json.dumps(doc_dict) + "\n")

    print("Number of instances processed:", instances_processed)
    print(output_path)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    minimize_split(input_dir, output_dir)
