import torch.nn as nn
from transformers import LongformerTokenizerFast, AutoModel
import torch


class BaseDocEncoder(nn.Module):
    def __init__(self, model_size='base', finetune=False, max_training_segments=4, device="cuda", **kwargs):
        super(BaseDocEncoder, self).__init__()
        self.device = device
        self.finetune = finetune

        if self.finetune:
            print("\nFinetuning the document encoder\n")

        self.max_training_segments = max_training_segments

        gradient_checkpointing = False
        if finetune:
            if torch.cuda.is_available():
                memory_in_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                if memory_in_gb < 40:
                    gradient_checkpointing = True
                gradient_checkpointing = False

        self.lm_encoder = AutoModel.from_pretrained(
            f"allenai/longformer-{model_size}-4096", output_hidden_states=False,
            gradient_checkpointing=gradient_checkpointing)

        if not self.finetune:
            for param in self.lm_encoder.parameters():
                # Don't update BERT params
                param.requires_grad = False

        self.hsize = self.lm_encoder.config.hidden_size

    def forward(self, example):
        return
