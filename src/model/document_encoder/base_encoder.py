import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch


class BaseDocEncoder(nn.Module):
    def __init__(self, config):
        super(BaseDocEncoder, self).__init__()

        gradient_checkpointing = False
        if config.finetune:
            gradient_checkpointing = True
            if torch.cuda.is_available():
                memory_in_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                if memory_in_gb > 40:
                    gradient_checkpointing = False

        model_str = config.transformer.model_str

        self.lm_encoder = AutoModel.from_pretrained(
            model_str, output_hidden_states=False,
            gradient_checkpointing=gradient_checkpointing)

        self.tokenizer = AutoTokenizer.from_pretrained(model_str)
        if config.add_speaker_tokens:
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': [config.SPEAKER_START, config.SPEAKER_END]
            })

            self.lm_encoder.resize_token_embeddings(len(self.tokenizer))

        if not config.finetune:
            for param in self.lm_encoder.parameters():
                # Don't update encoder params
                param.requires_grad = False

        self.hidden_size = self.lm_encoder.config.hidden_size

    def get_tokenizer(self):
        return self.tokenizer

    def to_add_speaker_tokens(self):
        return self.add_speaker_tokens

    def forward(self, example):
        return
