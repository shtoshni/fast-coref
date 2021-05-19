from transformers import LongformerTokenizerFast, AutoTokenizer


def get_tokenizer(model_str):
    if 'longformer' in model_str:
        tokenizer = LongformerTokenizerFast.from_pretrained(model_str, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_str)

    return tokenizer
