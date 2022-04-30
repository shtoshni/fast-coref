import collections

import re
import os
import json

from coref_utils import conll
from os import path
from data_processing.utils import (
    split_into_segments,
    flatten,
    get_sentence_map,
    parse_args,
    normalize_word,
    BaseDocumentState,
)
from data_processing.process_ontonotes import OntoNotesDocumentState


class DocumentState(OntoNotesDocumentState):
    def __init__(self, key):
        super().__init__(key)
        self.clusters = collections.defaultdict(list)

    def finalize(self):
        self.final_processing()
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "clusters": self.merged_clusters,
            "sentence_map": self.sentence_map,
            "subtoken_map": self.subtoken_map,
        }


def get_document(document_lines, tokenizer, segment_len):
    document_state = DocumentState(document_lines[0])
    word_idx = -1
    for line in document_lines[1]:
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12
            if len(row) == 12:
                row.append("-")
            word_idx += 1
            word = normalize_word(row[3])
            subtokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            document_state.tokens.append(word)
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if sidx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
        else:
            document_state.sentence_end[-1] = True

    split_into_segments(
        document_state,
        segment_len,
        document_state.sentence_end,
        document_state.token_end,
    )
    document = document_state.finalize()
    return document


def minimize_partition(
    split, cross_val_split, tokenizer, seg_len, input_dir, output_dir
):
    input_path = path.join(input_dir, "{}/{}.conll".format(cross_val_split, split))
    output_path = path.join(
        output_dir, "{}/{}.{}.jsonlines".format(cross_val_split, split, seg_len)
    )
    count = 0
    print("Minimizing {}".format(input_path))
    documents = []
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(
                    begin_document_match.group(1), begin_document_match.group(2)
                )
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)
    with open(output_path, "w") as output_file:
        for document_lines in documents:
            document = get_document(document_lines, tokenizer, seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(args):
    for cross_val_split in range(10):
        # Create cross validation output dir
        cross_val_dir = path.join(args.output_dir, str(cross_val_split))
        if not path.exists(cross_val_dir):
            os.makedirs(cross_val_dir)

        for split in ["dev", "test", "train"]:
            minimize_partition(
                split,
                cross_val_split,
                args.tokenizer,
                args.seg_len,
                args.input_dir,
                args.output_dir,
            )


if __name__ == "__main__":
    minimize_split(parse_args())
