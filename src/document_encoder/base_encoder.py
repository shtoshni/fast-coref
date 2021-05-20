import torch.nn as nn
from transformers import LongformerModel, AutoTokenizer
import torch

from auto_memory_model.constants import SPEAKER_START, SPEAKER_END


class BaseDocEncoder(nn.Module):
    def __init__(self, model_size='base', finetune=False, device="cuda", add_speaker_tokens=False, **kwargs):
        super(BaseDocEncoder, self).__init__()
        self.device = device
        self.finetune = finetune
        self.add_speaker_tokens = add_speaker_tokens

        gradient_checkpointing = False
        if finetune and self.training:
            gradient_checkpointing = True
            if torch.cuda.is_available():
                memory_in_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                if memory_in_gb > 40:
                    gradient_checkpointing = False

            # print(f"Gradient Checkpointing: {gradient_checkpointing}\n")

        self.lm_encoder = LongformerModel.from_pretrained(
            f"allenai/longformer-{model_size}-4096", output_hidden_states=False,
            gradient_checkpointing=gradient_checkpointing).to(device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-large-4096')
        if add_speaker_tokens:
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': [SPEAKER_START, SPEAKER_END]
            })

            self.lm_encoder.resize_token_embeddings(len(self.tokenizer))

        if not self.finetune:
            for param in self.lm_encoder.parameters():
                # Don't update encoder params
                param.requires_grad = False

        self.hsize = self.lm_encoder.config.hidden_size

    def get_tokenizer(self):
        return self.tokenizer

    def to_add_speaker_tokens(self):
        return self.add_speaker_tokens

    def forward(self, example):
        return
