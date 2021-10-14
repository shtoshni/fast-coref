import xml.etree.ElementTree as ET
import json
import numpy as np

from os import path
from data_processing.utils import flatten, BaseDocumentState, parse_args

TOTAL_INSTANCES = 273


class DocumentState(BaseDocumentState):
	def __init__(self, key):
		super().__init__(key)

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


def minimize_split(args, split="test"):
	tokenizer = args.tokenizer

	input_path = path.join(args.input_dir, "WSCollection.xml")
	output_path = path.join(args.output_dir, "test.jsonlines".format(split))
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
		num_tokens_list = []
		ment_len_list = []

		for elem in list(root)[:TOTAL_INSTANCES]:
			for children in list(elem.iter('txt1')):
				prefix = children.text.strip().replace('\n', ' ')
				prefixes.append(prefix)

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
						answer_boundaries.append([found, found + len(span_tokens) - 1])
						break

				if found == -1:
					print(text, answer)
					not_found_count += 1

			if len(answer_boundaries) == 2:
				document = DocumentState(f'wsc_{idx}')
				num_tokens_list.append(len(text.split()))

				ment_len_list.extend([1, len(answer1.split()), len(answer2.split())])

				correct_answer = correct_answers[idx]
				assert (correct_answer in ['A', 'B'])

				if correct_answer == 'A':
					document.a_label = 1
				else:
					document.b_label = 1

				document.tokens = word_list
				document.segments = [tokenizer.convert_tokens_to_ids(word_list)]
				document.subtoken_map = list(range(len(word_list)))

				document.pronoun_span = pronoun_boundary
				document.a_span = answer_boundaries[0]
				document.b_span = answer_boundaries[1]

				doc_dict = document.finalize()
				instances_processed += 1
				out_f.write(json.dumps(doc_dict) + "\n")

	print("Number of instances processed:", instances_processed)
	print(f"Number of tokens per doc: {np.mean(num_tokens_list):.1f}")
	print(f"Avg, mention length: {np.mean(ment_len_list):.1f}")
	print(output_path)


if __name__ == "__main__":
	minimize_split(parse_args())
