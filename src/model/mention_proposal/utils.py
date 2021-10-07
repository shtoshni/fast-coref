import torch


def sort_mentions(ment_starts, ment_ends, return_sorted_indices=False):
	sort_scores = ment_starts + 1e-5 * ment_starts
	_, sorted_indices = torch.sort(sort_scores, 0)

	output = (ment_starts[sorted_indices], ment_ends[sorted_indices])
	if return_sorted_indices:
		output += (sorted_indices,)

	return output


def transform_gold_mentions(self, document):
	"Transform gold mentions given the document."
	mentions = []
	for cluster in document["clusters"]:
		for ment_start, ment_end in cluster:
			mentions.append((ment_start, ment_end))

	if len(mentions):
		topk_starts, topk_ends = zip(*mentions)
	else:
		raise ValueError

	topk_starts = torch.tensor(topk_starts, device=self.device)
	topk_ends = torch.tensor(topk_ends, device=self.device)

	topk_starts, topk_ends = sort_mentions(topk_starts, topk_ends)

	output_dict = {
		'ment_starts': topk_starts, 'ment_ends': topk_ends,
		# Fake mention score
		'pred_scores': torch.tensor([1.0] * len(mentions), device=self.device)
	}

	return output_dict
