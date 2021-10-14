import torch
from typing import Tuple, Any


def sort_mentions(
				ment_starts: torch.Tensor, ment_ends: torch.Tensor, return_sorted_indices=False) -> Tuple:
	sort_scores = ment_starts + 1e-5 * ment_starts
	_, sorted_indices = torch.sort(sort_scores, 0)

	output: Tuple = (ment_starts[sorted_indices], ment_ends[sorted_indices])
	if return_sorted_indices:
		output = output + (sorted_indices,)

	return output

