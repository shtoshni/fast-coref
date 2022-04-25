from transformers import LongformerTokenizerFast, AutoTokenizer, PreTrainedTokenizerFast
from transformers import LongformerSelfAttention, BertForMaskedLM


def get_tokenizer(model_str: str) -> PreTrainedTokenizerFast:
    if 'longformer' in model_str:
        tokenizer = LongformerTokenizerFast.from_pretrained(model_str, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=True)

    return tokenizer


class BertLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        return super().forward(
            hidden_states, attention_mask=attention_mask)

    def self(self, hidden_states, attention_mask=None, head_mask=None, **kwargs):
        return super().forward(
            hidden_states, attention_mask=attention_mask)


class BertLong(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.bert.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with
            # `LongformerSelfAttention`
            layer.attention = BertLongSelfAttention(config, layer_id=i)