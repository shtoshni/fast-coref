import re
import json

from coref_utils import conll
from os import path
from data_processing.utils import split_into_segments, flatten, normalize_word, get_sentence_map, \
	parse_args, BaseDocumentState


class DocumentState(BaseDocumentState):
	def __init__(self, key):
		super().__init__(key)

	def finalize(self):
		# finalized: segments, segment_subtoken_map
		# populate speakers from info
		subtoken_idx = 0
		for segment in self.segment_info:
			speakers = []
			for i, tok_info in enumerate(segment):
				if tok_info is None and (i == 0 or i == len(segment) - 1):
					speakers.append('[SPL]')
				elif tok_info is None:
					speakers.append(speakers[-1])
				else:
					speakers.append(tok_info[9])
					if tok_info[4] == 'PRP':
						self.pronouns.append(subtoken_idx)
				subtoken_idx += 1
			self.speakers += [speakers]
		# populate sentence map

		# populate clusters
		first_subtoken_index = -1
		for seg_idx, segment in enumerate(self.segment_info):
			for i, tok_info in enumerate(segment):
				first_subtoken_index += 1
				coref = tok_info[-2] if tok_info is not None else '-'
				if coref != "-":
					last_subtoken_index = first_subtoken_index + \
					                      tok_info[-1] - 1
					for part in coref.split("|"):
						if part[0] == "(":
							if part[-1] == ")":
								cluster_id = int(part[1:-1])
								self.clusters[cluster_id].append(
									(first_subtoken_index, last_subtoken_index))
							else:
								cluster_id = int(part[1:])
								self.coref_stacks[cluster_id].append(
									first_subtoken_index)
						else:
							cluster_id = int(part[:-1])
							start = self.coref_stacks[cluster_id].pop()
							self.clusters[cluster_id].append((start, last_subtoken_index))

		# merge clusters
		merged_clusters = []
		for c1 in self.clusters.values():
			existing = None
			for m in c1:
				for c2 in merged_clusters:
					if m in c2:
						existing = c2
						break
				if existing is not None:
					break
			if existing is not None:
				print("Merging clusters (shouldn't happen very often.)")
				existing.update(c1)
			else:
				merged_clusters.append(set(c1))
		merged_clusters = [list(c) for c in merged_clusters]
		all_mentions = flatten(merged_clusters)
		sentence_map = get_sentence_map(self.segments, self.sentence_end)
		subtoken_map = flatten(self.segment_subtoken_map)
		assert len(all_mentions) == len(set(all_mentions))
		num_words = len(flatten(self.segments))
		assert num_words == len(flatten(self.speakers))
		assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
		assert num_words == len(sentence_map), (num_words, len(sentence_map))
		return {
			"doc_key": self.doc_key,
			"sentences": self.segments,
			"clusters": merged_clusters,
			'sentence_map': sentence_map,
			"subtoken_map": subtoken_map,
		}


def get_document(document_lines, tokenizer, segment_len):
	document_state = DocumentState(document_lines[0])
	word_idx = -1
	for line in document_lines[1]:
		row = line.split()
		sentence_end = len(row) == 0
		if not sentence_end:
			assert len(row) == 12
			word_idx += 1
			word = normalize_word(row[3])
			subtokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
			document_state.tokens.append(word)
			document_state.token_end += ([False]
			                             * (len(subtokens) - 1)) + [True]
			for sidx, subtoken in enumerate(subtokens):
				document_state.subtokens.append(subtoken)
				info = None if sidx != 0 else (row + [len(subtokens)])
				document_state.info.append(info)
				document_state.sentence_end.append(False)
				document_state.subtoken_map.append(word_idx)
		else:
			document_state.sentence_end[-1] = True

	split_into_segments(document_state, segment_len, document_state.sentence_end, document_state.token_end)
	document = document_state.finalize()
	return document


def minimize_split(args, split="test"):
	tokenizer = args.tokenizer

	input_path = path.join(args.input_dir, "{}.conll".format(split))
	output_path = path.join(args.output_dir, "{}.{}.jsonlines".format(split, args.seg_len))
	count = 0
	print("Minimizing {}".format(input_path))
	documents = []
	with open(input_path, "r") as input_file:
		for line in input_file.readlines():
			begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
			if begin_document_match:
				doc_key = conll.get_doc_key(
					begin_document_match.group(1), begin_document_match.group(2))
				documents.append((doc_key, []))
			elif line.startswith("#end document"):
				continue
			else:
				documents[-1][1].append(line)
	with open(output_path, "w") as output_file:
		for document_lines in documents:
			document = get_document(document_lines, tokenizer, args.seg_len)
			output_file.write(json.dumps(document))
			output_file.write("\n")
			count += 1
	print("Wrote {} documents to {}".format(count, output_path))


if __name__ == "__main__":
	minimize_split(parse_args())
