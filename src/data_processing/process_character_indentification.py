from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import os
import sys
import json
import collections
from collections import defaultdict

from coref_utils import conll
from os import path
from transformers import LongformerTokenizerFast
from auto_memory_model.constants import SPEAKER_START, SPEAKER_END



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
        self.clusters = []
        self.cluster_str = []
        self.segment_info = []

    def finalize(self):
        all_mentions = flatten(self.clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "cluster_str": self.cluster_str,
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


def process_speaker(speaker_list):
    return ' and '.join(speaker_list)


def get_document(instance, tokenizer, segment_len, add_speaker=False):
    document_state = DocumentState(instance["scene_id"])
    general_counter = 0

    clusters = defaultdict(list)
    token_counter = 0
    for utterance in instance['utterances']:
        speaker = tuple(sorted(utterance['speakers']))
        if add_speaker:
            # Insert speaker tokens
            speaker_str = process_speaker(speaker)
            document_state.tokens.append(SPEAKER_START)
            document_state.tokens.extend(tokenizer.tokenize(speaker_str))
            document_state.tokens.append(SPEAKER_END)

            speaker_subtoken_ids = []
            speaker_subtoken_ids.extend(tokenizer.convert_tokens_to_ids([SPEAKER_START]))
            speaker_subtoken_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(speaker_str))),
            speaker_subtoken_ids.extend(tokenizer.convert_tokens_to_ids([SPEAKER_END]))

            document_state.token_end += ([False] * (len(speaker_subtoken_ids) - 1)) + [True]
            for sidx, subtoken in enumerate(speaker_subtoken_ids):
                document_state.subtokens.append(subtoken)
                # document_state.tokens.append(subtoken)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(token_counter)

            token_counter += 1

        utterance_clusters = defaultdict(list)
        for idx, (sent, per_sent_entities) in enumerate(
                zip(utterance['tokens'], utterance['character_entities'])):
            sentence_token_map = {}
            for token_idx, token in enumerate(sent):
                sentence_token_map[token_idx] = [len(document_state.subtokens)]
                subtokens = tokenizer.tokenize(token)
                subtoken_ids = tokenizer.convert_tokens_to_ids(subtokens)
                document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
                for sidx, (subtoken, subtoken_id) in enumerate(zip(subtokens, subtoken_ids)):
                    document_state.subtokens.append(subtoken_id)
                    document_state.tokens.append(subtoken)
                    document_state.sentence_end.append(False)
                    document_state.subtoken_map.append(token_counter)

                sentence_token_map[token_idx].append(len(document_state.subtokens))
                token_counter += 1

            if len(document_state.sentence_end):
                document_state.sentence_end[-1] = True

            for entity in per_sent_entities:
                characters = tuple(sorted(entity[2:]))
                token_start, token_end = entity[:2]

                span_start = sentence_token_map[token_start][0]
                span_end = sentence_token_map[token_end - 1][1] - 1
                utterance_clusters[characters].append((span_start, span_end))

        for character in utterance_clusters:
            if character != '#GENERAL#':
                clusters[character].extend(utterance_clusters[character])
            else:
                clusters[character + str(general_counter)] = utterance_clusters[character]
                general_counter += 1

    for entity, cluster in clusters.items():
        cluster_str = []
        for ment_start, ment_end in cluster:
            cluster_str.append(tokenizer.convert_tokens_to_string(document_state.tokens[ment_start: ment_end + 1]))

        document_state.clusters.append(cluster)
        document_state.cluster_str.append(cluster_str)

    split_into_segments(document_state, segment_len, document_state.sentence_end, document_state.token_end)
    document = document_state.finalize()
    return document


def minimize_partition(split, tokenizer, seg_len, input_dir, output_dir, add_speaker=False):
    split_to_src_doc = {'train': 'trn', 'test': 'tst', 'dev': 'dev'}
    input_path = path.join(input_dir, "character-identification-{}.json".format(split_to_src_doc[split]))
    output_path = path.join(output_dir, "{}.{}.jsonlines".format(split, seg_len))

    count = 0
    print("Minimizing {}".format(input_path))
    with open(input_path, "r") as input_f, open(output_path, "w") as output_w:
        data = json.load(input_f)
        for episode in data['episodes']:
            for scene in episode['scenes']:
                document = get_document(scene, tokenizer, segment_len=seg_len, add_speaker=add_speaker)
                output_w.write(json.dumps(document))
                output_w.write("\n")
                count += 1

        print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(seg_len, input_dir, output_dir, add_speaker=False):
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096', add_prefix_space=True)
    if add_speaker:
        tokenizer.add_special_tokens({
            'additional_special_tokens': [SPEAKER_START, SPEAKER_END]
        })

    for split in ["dev", "test", "train"]:
        minimize_partition(split, tokenizer, seg_len, input_dir, output_dir, add_speaker=add_speaker)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', type=str)
    parser.add_argument('-output_dir', type=str)
    parser.add_argument('-add_speaker', default=False, action="store_true")

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    for seg_len in [2048, 4096]:
        minimize_split(seg_len, args.input_dir, args.output_dir, add_speaker=args.add_speaker)