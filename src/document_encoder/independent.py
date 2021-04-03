import torch
import torch.nn as nn
from os import path
import random
from pytorch_utils.utils import get_sequence_mask, get_span_mask
from document_encoder.base_encoder import BaseDocEncoder


class IndependentDocEncoder(BaseDocEncoder):
    def __init__(self, **kwargs):
        super(IndependentDocEncoder, self).__init__(**kwargs)

    def encode_doc(self, document, text_length_list):
        """
        Encode chunks of a document.
        batch_excerpt: C x L where C is number of chunks padded upto max length of L
        text_length_list: list of length of chunks (length C)
        """
        num_chunks = len(text_length_list)
        attn_mask = get_sequence_mask(torch.tensor(text_length_list, device=self.device))

        if not self.finetune:
            with torch.no_grad():
                outputs = self.lm_encoder(document, attention_mask=attn_mask)  # C x L x E
        else:
            outputs = self.lm_encoder(document, attention_mask=attn_mask)  # C x L x E

        encoded_repr = outputs[0]

        unpadded_encoded_output = []
        for i in range(num_chunks):
            unpadded_encoded_output.append(
                encoded_repr[i, 1:text_length_list[i]-1, :])

        encoded_output = torch.cat(unpadded_encoded_output, dim=0)
        encoded_output = encoded_output

        return encoded_output

    def tensorize_example(self, example):
        if self.training and self.max_training_segments is not None:
            example = self.truncate_document(example)
        sentences = example["sentences"]

        sentences = [([self.tokenizer.cls_token] + sent + [self.tokenizer.sep_token]) for sent in sentences]
        sent_len_list = [len(sent) for sent in sentences]
        max_sent_len = max(sent_len_list)
        padded_sent = [self.tokenizer.convert_tokens_to_ids(sent)
                       + [self.pad_token] * (max_sent_len - len(sent))
                       for sent in sentences]
        doc_tens = torch.tensor(padded_sent, device=self.device)
        return doc_tens, sent_len_list

    def truncate_document(self, example):
        sentences = example["sentences"]
        num_sentences = len(example["sentences"])

        if num_sentences > self.max_training_segments:
            sentence_offset = random.randint(0, num_sentences - self.max_training_segments)
            word_offset = sum([len(sent) for sent in sentences[:sentence_offset]])
            sentences = sentences[sentence_offset: sentence_offset + self.max_training_segments]
            num_words = sum([len(sent) for sent in sentences])
            sentence_map = example["sentence_map"][word_offset: word_offset + num_words]

            clusters = []
            for orig_cluster in example["clusters"]:
                cluster = []
                for ment_start, ment_end in orig_cluster:
                    if ment_end >= word_offset and ment_start < word_offset + num_words:
                        cluster.append((ment_start - word_offset, ment_end - word_offset))

                if len(cluster):
                    clusters.append(cluster)

            example["sentences"] = sentences
            example["clusters"] = clusters
            example["sentence_map"] = sentence_map

            return example
        else:
            return example

    def forward(self, example):
        return self.encode_doc(*self.tensorize_example(example))
