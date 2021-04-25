import torch.nn as nn
from transformers import LongformerModel
import torch


class BaseDocEncoder(nn.Module):
    def __init__(self, model_size='base', finetune=False, device="cuda", **kwargs):
        super(BaseDocEncoder, self).__init__()
        self.device = device
        self.finetune = finetune

        if self.finetune:
            print("\nFinetuning the document encoder\n")

        gradient_checkpointing = False
        if finetune and self.training:
            gradient_checkpointing = True
            if torch.cuda.is_available():
                memory_in_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                if memory_in_gb > 40:
                    gradient_checkpointing = False

            print(f"Gradient Checkpointing: {gradient_checkpointing}\n")

        self.lm_encoder = LongformerModel.from_pretrained(
            f"allenai/longformer-{model_size}-4096", output_hidden_states=False,
            gradient_checkpointing=gradient_checkpointing)

        if not self.finetune:
            for param in self.lm_encoder.parameters():
                # Don't update encoder params
                param.requires_grad = False

        self.hsize = self.lm_encoder.config.hidden_size

    def forward(self, example):
        return
