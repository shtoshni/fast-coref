import json
from collections import defaultdict

from os import path
from data_processing.constants import SPEAKER_START, SPEAKER_END
from data_processing.utils import split_into_segments, parse_args
from data_processing.process_preco import PrecoDocumentState


class DocumentState(PrecoDocumentState):
    def __init__(self, key):
        super().__init__(key)

    def finalize(self):
        self.final_process()
        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "cluster_str": self.cluster_str,
            "sentences": self.segments,
            "clusters": self.clusters,
            "sentence_map": self.sentence_map,
            "subtoken_map": self.subtoken_map,
        }


def process_speaker(speaker_list):
    return " and ".join(speaker_list)


def get_document(instance, tokenizer, segment_len, add_speaker=False):
    document_state = DocumentState(instance["scene_id"])
    general_counter = 0

    clusters = defaultdict(list)
    token_counter = 0
    for utterance in instance["utterances"]:
        speaker = tuple(sorted(utterance["speakers"]))
        if add_speaker:
            # Insert speaker tokens
            speaker_str = process_speaker(speaker)
            document_state.tokens.append(SPEAKER_START)
            document_state.tokens.extend(tokenizer.tokenize(speaker_str))
            document_state.tokens.append(SPEAKER_END)

            speaker_subtoken_ids = []
            speaker_subtoken_ids.extend(
                tokenizer.convert_tokens_to_ids([SPEAKER_START])
            )
            speaker_subtoken_ids.extend(
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(speaker_str))
            ),
            speaker_subtoken_ids.extend(tokenizer.convert_tokens_to_ids([SPEAKER_END]))

            document_state.token_end += ([False] * (len(speaker_subtoken_ids) - 1)) + [
                True
            ]
            for sidx, subtoken in enumerate(speaker_subtoken_ids):
                document_state.subtokens.append(subtoken)
                # document_state.tokens.append(subtoken)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(token_counter)

            token_counter += 1

        utterance_clusters = defaultdict(list)
        for idx, (sent, per_sent_entities) in enumerate(
            zip(utterance["tokens"], utterance["character_entities"])
        ):
            sentence_token_map = {}
            for token_idx, token in enumerate(sent):
                sentence_token_map[token_idx] = [len(document_state.subtokens)]
                subtokens = tokenizer.tokenize(token)
                subtoken_ids = tokenizer.convert_tokens_to_ids(subtokens)
                document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
                for sidx, (subtoken, subtoken_id) in enumerate(
                    zip(subtokens, subtoken_ids)
                ):
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
            if character != "#GENERAL#":
                clusters[character].extend(utterance_clusters[character])
            else:
                clusters[character + str(general_counter)] = utterance_clusters[
                    character
                ]
                general_counter += 1

    for entity, cluster in clusters.items():
        cluster_str = []
        for ment_start, ment_end in cluster:
            cluster_str.append(
                tokenizer.convert_tokens_to_string(
                    document_state.tokens[ment_start : ment_end + 1]
                )
            )

        document_state.clusters.append(cluster)
        document_state.cluster_str.append(cluster_str)

    split_into_segments(
        document_state,
        segment_len,
        document_state.sentence_end,
        document_state.token_end,
    )
    document = document_state.finalize()
    return document


def minimize_partition(split, args):
    split_to_src_doc = {"train": "trn", "test": "tst", "dev": "dev"}
    input_path = path.join(
        args.input_dir,
        "character-identification-{}.json".format(split_to_src_doc[split]),
    )
    output_path = path.join(
        args.output_dir, "{}.{}.jsonlines".format(split, args.seg_len)
    )

    count = 0
    print("Minimizing {}".format(input_path))
    with open(input_path, "r") as input_f, open(output_path, "w") as output_w:
        data = json.load(input_f)
        for episode in data["episodes"]:
            for scene in episode["scenes"]:
                document = get_document(
                    scene,
                    args.tokenizer,
                    segment_len=args.seg_len,
                    add_speaker=args.add_speaker,
                )
                output_w.write(json.dumps(document))
                output_w.write("\n")
                count += 1

        print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(args):
    tokenizer = args.tokenizer
    if args.add_speaker:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [SPEAKER_START, SPEAKER_END]}
        )

    for split in ["dev", "test", "train"]:
        minimize_partition(split, args)


if __name__ == "__main__":
    minimize_split(parse_args())
