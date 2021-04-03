import torch
import random
from pytorch_utils.utils import get_sequence_mask
from document_encoder.base_encoder import BaseDocEncoder


class OverlapDocEncoder(BaseDocEncoder):
    def __init__(self, **kwargs):
        super(OverlapDocEncoder, self).__init__(**kwargs)

    def encode_doc(self, example):
        """
        Encode chunks of a document.
        batch_excerpt: C x L where C is number of chunks padded upto max length of L
        text_length_list: list of length of chunks (length C)
        """

        doc_tens = example["padded_sent"].to(self.device)
        sent_len_list = example["sent_len_list"]
        start_indices = example["start_indices"]
        end_indices = example["end_indices"]
        num_chunks = sent_len_list.shape[0]
        attn_mask = get_sequence_mask(sent_len_list.to(self.device))

        if not self.finetune:
            with torch.no_grad():
                outputs = self.lm_encoder(doc_tens, attention_mask=attn_mask)  # C x L x E
        else:
            outputs = self.lm_encoder(doc_tens, attention_mask=attn_mask)  # C x L x E

        encoded_repr = outputs[0]

        unpadded_encoded_output = []
        offset = 1  # for cls_token which was not accounted during segmentation
        for i in range(num_chunks):
            unpadded_encoded_output.append(
                encoded_repr[i, offset + start_indices[i]: offset + end_indices[i], :])

        encoded_output = torch.cat(unpadded_encoded_output, dim=0)
        encoded_output = encoded_output
        return encoded_output

    def forward(self, example):
        return self.encode_doc(example)
