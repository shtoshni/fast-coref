import glob
import json

import xml.etree.ElementTree as ET
from collections import defaultdict
from os import path
from data_processing.utils import split_into_segments, parse_args
from data_processing.process_preco import PrecoDocumentState


class DocumentState(PrecoDocumentState):
    def __init__(self, key):
        super().__init__(key)

    def finalize(self):
        self.final_process()
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "str_doc": self.tokens,
            "clusters": self.clusters,
            "sentence_map": self.sentence_map,
            "subtoken_map": self.subtoken_map,
        }


def get_document(text_file, xml_file, tokenizer, segment_len):
    document_state = DocumentState(path.basename(text_file))
    doc_word_idx = 0

    sentence_word_map = {}
    for line in open(text_file):
        word = line.strip()
        if word != "":
            doc_word_idx += 1
            sentence_word_map[doc_word_idx] = [len(document_state.subtokens)]

            subtokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                document_state.tokens.append(tokenizer.convert_ids_to_tokens(subtoken))
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(doc_word_idx)

            sentence_word_map[doc_word_idx].append(len(document_state.subtokens))
            sentence_word_map[doc_word_idx].append(word)
        else:
            document_state.sentence_end[-1] = True

    # Map WikiCoref clusters
    tree = ET.parse(xml_file)
    root = tree.getroot()

    coref_class_to_spans = defaultdict(list)
    # uniq_spans = set()
    for elem in list(root):
        if elem.get("coreftype") == "ident":
            span = elem.get("span")
            coref_class = elem.get("coref_class")

            # Find span boundaries
            word_span_start, word_span_end = span.split("..")

            word_span_start = int(word_span_start.split("_")[1])
            span_start = sentence_word_map[word_span_start][0]

            word_span_end = int(word_span_end.split("_")[1])
            span_end = sentence_word_map[word_span_end][1] - 1
            coref_class_to_spans[coref_class].append((span_start, span_end))

    coref_class_list = list(coref_class_to_spans.keys())
    # Remove singletons
    for coref_class in coref_class_list:
        if len(coref_class_to_spans[coref_class]) == 1:
            # print(xml_file)
            # print(coref_class, coref_class_to_spans[coref_class])
            # span_start, span_end = coref_class_to_spans[coref_class][0]
            # print(tokenizer.convert_tokens_to_string(document_state.tokens[span_start: span_end + 1]))
            del coref_class_to_spans[coref_class]

    document_state.clusters = list(coref_class_to_spans.values())

    split_into_segments(
        document_state,
        segment_len,
        document_state.sentence_end,
        document_state.token_end,
    )
    document = document_state.finalize()
    return document


def minimize_split(args, split="test"):
    tokenizer = args.tokenizer

    annotation_path = path.join(args.input_dir, "Annotation")
    if path.exists(annotation_path):
        args.input_dir = annotation_path

    text_files = glob.glob(path.join(args.input_dir, "*/*.txt"))
    xml_files = []
    for text_file in text_files:
        markable_dir = path.join(path.dirname(text_file), "Markables")
        ontonotes_file = glob.glob(path.join(markable_dir, "*_OntoNotes*.xml"))[0]
        xml_files.append(ontonotes_file)

    output_path = path.join(
        args.output_dir, "{}.{}.jsonlines".format(split, args.seg_len)
    )
    count = 0
    with open(output_path, "w") as output_file:
        for (text_file, xml_file) in zip(text_files, xml_files):
            document = get_document(text_file, xml_file, tokenizer, args.seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


if __name__ == "__main__":
    minimize_split(parse_args())
