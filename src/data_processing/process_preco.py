from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import os
import sys
import json
import collections

from coref_utils import conll
from os import path
from transformers import LongformerTokenizerFast


class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.pronouns = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)
        self.segment_info = []

    def finalize(self):
        all_mentions = flatten(self.clusters )
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "clusters": self.clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
        }


def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def flatten(l):
  return [item for sublist in l for item in sublist]


def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
    current = 0
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
        document_state.segments.append(
            document_state.subtokens[current:end + 1])
        subtoken_map = document_state.subtoken_map[current: end + 1]
        document_state.segment_subtoken_map.append(subtoken_map)
        info = document_state.info[current: end + 1]
        document_state.segment_info.append(info)
        current = end + 1


def get_sentence_map(segments, sentence_end):
    current = 0
    sent_map = []
    sent_end_idx = 0
    assert len(sentence_end) == sum([len(s) for s in segments])
    for segment in segments:
        for i in range(len(segment)):
            sent_map.append(current)
            current += int(sentence_end[sent_end_idx])
            sent_end_idx += 1
    return sent_map


def get_document(instance, tokenizer, segment_len):
    document_state = DocumentState(instance["id"])
    doc_word_idx = -1

    sentence_word_map = {}
    for sentence_idx, sentence in enumerate(instance["sentences"]):
        sentence_word_map[sentence_idx] = {}
        for word_idx, word in enumerate(sentence):
            doc_word_idx += 1
            sentence_word_map[sentence_idx][word_idx] = [len(document_state.subtokens)]
            subtokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            # subtokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))

            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(doc_word_idx)

            sentence_word_map[sentence_idx][word_idx].append(len(document_state.subtokens))

        document_state.sentence_end[-1] = True

    # Map preco clusters
    mapped_clusters = []
    for cluster in instance["mention_clusters"]:
        cur_cluster = []
        for (sent_idx, word_start, word_end) in cluster:
            # assert (sentence_word_map[sent_idx][word_start][0] < len(document_state.sentence_end))
            span_start = sentence_word_map[sent_idx][word_start][0]
            span_end = sentence_word_map[sent_idx][word_end - 1][1] - 1
            tokens = tokenizer.convert_ids_to_tokens(document_state.subtokens[span_start: span_end + 1])
            # mention_str = tokenizer.convert_tokens_to_string(tokens)
            # cur_cluster.append((span_start, span_end, mention_str))
            cur_cluster.append((span_start, span_end))
        mapped_clusters.append(sorted(cur_cluster, key=lambda x: x[0]))

    document_state.clusters = mapped_clusters

    split_into_segments(document_state, segment_len, document_state.sentence_end, document_state.token_end)
    document = document_state.finalize()
    return document


def minimize_partition(split, tokenizer, seg_len, input_dir, output_dir):
    input_path = path.join(input_dir, "{}.jsonl".format(split))
    output_path = path.join(output_dir, "{}.{}.jsonlines".format(split, seg_len))
    count = 0
    print("Minimizing {}".format(input_path))
    with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
        for line in input_file.readlines():
            instance = json.loads(line.strip())
            document = get_document(instance, tokenizer, seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(seg_len, input_dir, output_dir):
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096', add_prefix_space=True)
    # Create cross validation output dir

    minimize_partition("dev", tokenizer, seg_len, input_dir, output_dir)
    minimize_partition("test", tokenizer, seg_len, input_dir, output_dir)
    minimize_partition("train", tokenizer, seg_len, input_dir, output_dir)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for seg_len in [2048, 4096]:
        minimize_split(seg_len, input_dir, output_dir)