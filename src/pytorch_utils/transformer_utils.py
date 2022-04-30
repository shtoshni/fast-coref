from transformers import LongformerTokenizerFast, AutoTokenizer, PreTrainedTokenizerFast


def get_tokenizer(model_str: str) -> PreTrainedTokenizerFast:
    if "longformer" in model_str:
        tokenizer = LongformerTokenizerFast.from_pretrained(
            model_str, add_prefix_space=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=True)

    return tokenizer
