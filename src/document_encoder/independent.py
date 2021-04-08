import torch
import torch.nn as nn
from os import path
import random
from pytorch_utils.utils import get_sequence_mask, get_span_mask
from document_encoder.base_encoder import BaseDocEncoder


class IndependentDocEncoder(BaseDocEncoder):
    def __init__(self, **kwargs):
        super(IndependentDocEncoder, self).__init__(**kwargs)

    def forward(self, instance):
        """
        Encode chunks of a document.
        batch_excerpt: C x L where C is number of chunks padded upto max length of L
        text_length_list: list of length of chunks (length C)
        """
        doc_tens = instance["padded_sent"]
        if isinstance(doc_tens, list):
            doc_tens = torch.tensor(doc_tens, device=self.device)
        else:
            doc_tens = doc_tens.to(self.device)

        sent_len_list = instance["sent_len_list"]
        num_chunks = len(sent_len_list)
        if num_chunks == 1:
            attn_mask = None
        else:
            attn_mask = get_sequence_mask(torch.tensor(sent_len_list, device=self.device))

        if not self.finetune:
            with torch.no_grad():
                outputs = self.lm_encoder(doc_tens, attention_mask=attn_mask)  # C x L x E
        else:
            outputs = self.lm_encoder(doc_tens, attention_mask=attn_mask)  # C x L x E

        encoded_repr = outputs[0]

        unpadded_encoded_output = []
        for idx, sent_len in enumerate(sent_len_list):
            unpadded_encoded_output.append(
                encoded_repr[idx, 1:sent_len-1, :])

        encoded_output = torch.cat(unpadded_encoded_output, dim=0)
        encoded_output = encoded_output

        return encoded_output
