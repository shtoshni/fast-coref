import json
from os import path

from data_processing.utils import (
    split_into_segments,
    flatten,
    get_sentence_map,
    parse_args,
    BaseDocumentState,
)


class PrecoDocumentState(BaseDocumentState):
    def __init__(self, key):
        super().__init__(key)

    def final_process(self):
        all_mentions = flatten(self.clusters)
        self.sentence_map = get_sentence_map(self.segments, self.sentence_end)
        self.subtoken_map = flatten(self.segment_subtoken_map)
        assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(self.subtoken_map), (num_words, len(self.subtoken_map))
        assert num_words == len(self.sentence_map), (num_words, len(self.sentence_map))

    def finalize(self):
        self.final_process()
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "clusters": self.clusters,
            "sentence_map": self.sentence_map,
            "subtoken_map": self.subtoken_map,
        }


def get_document(instance, tokenizer, segment_len):
    document_state = PrecoDocumentState(instance["id"])
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

            sentence_word_map[sentence_idx][word_idx].append(
                len(document_state.subtokens)
            )

        document_state.sentence_end[-1] = True

    # Map preco clusters
    mapped_clusters = []
    for cluster in instance["mention_clusters"]:
        cur_cluster = []
        for (sent_idx, word_start, word_end) in cluster:
            span_start = sentence_word_map[sent_idx][word_start][0]
            span_end = sentence_word_map[sent_idx][word_end - 1][1] - 1
            cur_cluster.append((span_start, span_end))
        mapped_clusters.append(sorted(cur_cluster, key=lambda x: x[0]))

    document_state.clusters = mapped_clusters

    split_into_segments(
        document_state,
        segment_len,
        document_state.sentence_end,
        document_state.token_end,
    )
    document = document_state.finalize()
    return document


def minimize_partition(split, tokenizer, args):
    input_path = path.join(args.input_dir, "{}.jsonl".format(split))
    output_path = path.join(
        args.output_dir, "{}.{}.jsonlines".format(split, args.seg_len)
    )
    count = 0
    print("Minimizing {}".format(input_path))
    with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
        for line in input_file.readlines():
            instance = json.loads(line.strip())
            document = get_document(instance, tokenizer, args.seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(args):
    tokenizer = args.tokenizer

    for split in ["dev", "test", "train"]:
        minimize_partition(split, tokenizer, args)


if __name__ == "__main__":
    minimize_split(parse_args())
