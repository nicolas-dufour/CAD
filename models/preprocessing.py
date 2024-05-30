import torch
import transformers
from torch import nn
from typing import Tuple


class TextPreprocessing(nn.Module):
    def __init__(
        self,
        tokenizer,
        model,
        drop_labels=False,
        input_key="text",
        output_key_root="text",
    ):
        super().__init__()
        self.cond_encoding = TextTokenizerPreprocess(
            tokenizer, input_key, output_key_root
        )
        self.cond_preprocess = TextEncoderPreprocess(
            model,
            input_ids_key=f"{output_key_root}_ids",
            input_mask_key=f"{output_key_root}_mask",
            output_key=f"{output_key_root}_embeddings",
        )
        self.drop_labels = drop_labels
        self.input_key = input_key
        self.output_key_root = output_key_root

    def forward(self, batch, training=True, device=None):
        if self.drop_labels:
            batch[f"{self.output_key_root}_embeddings"] = None
            batch[f"{self.output_key_root}_mask"] = None
            return batch
        batch = self.cond_encoding(batch)
        has_to_device = callable(getattr(self.cond_encoding, "to_device", None))
        if has_to_device:
            batch = self.cond_encoding.to_device(batch, device)
        batch = self.cond_preprocess(batch)
        return batch


class PrecomputedTextConditioning:
    def __init__(
        self,
        input_key="flan_t5_xl",
        output_key_root="text",
        drop_labels=False,
    ):
        self.input_key = input_key
        self.output_key_root = output_key_root
        self.drop_labels = drop_labels

    def __call__(self, batch, device=None):
        if self.drop_labels:
            batch[f"{self.output_key_root}_embeddings"] = None
            batch[f"{self.output_key_root}_mask"] = None
            return batch
        batch[f"{self.output_key_root}_embeddings"] = batch[
            f"{self.input_key}_embeddings"
        ]
        batch[f"{self.output_key_root}_mask"] = batch[f"{self.input_key}_mask"]
        return batch


class TextTokenizerPreprocess:
    def __init__(self, tokenizer, input_key="text", output_key_root="text"):
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key_root = output_key_root

    def __call__(self, batch):
        conditioning = batch[self.input_key]
        conditioning = self.tokenizer(
            conditioning,
            truncation=True,
            padding="longest",  # "max_length
            max_length=128,
            return_tensors="pt",
        )
        batch[f"{self.output_key_root}_mask"] = conditioning["attention_mask"].to(
            torch.bool
        )
        batch[f"{self.output_key_root}_ids"] = conditioning["input_ids"]
        return batch

    def to_device(self, batch, device):
        batch[f"{self.output_key_root}_ids"] = batch[f"{self.output_key_root}_ids"].to(
            device
        )
        batch[f"{self.output_key_root}_mask"] = batch[
            f"{self.output_key_root}_mask"
        ].to(device)
        return batch


class TextEncoderPreprocess(nn.Module):
    def __init__(
        self,
        text_model,
        input_ids_key="text_ids",
        input_mask_key="text_mask",
        output_key="text_embeddings",
    ):
        super().__init__()
        self.text_model = text_model
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.input_ids_key = input_ids_key
        self.input_mask_key = input_mask_key
        self.output_key = output_key

    def forward(self, batch):
        conditioning = {
            "input_ids": batch[f"{self.input_ids_key}"],
            "attention_mask": batch[f"{self.input_mask_key}"],
        }
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if isinstance(self.text_model, transformers.T5EncoderModel):
                    conditioning = self.text_model(
                        **conditioning
                    ).last_hidden_state.detach()
                elif isinstance(self.text_model, transformers.CLIPTextModel):
                    conditioning = (
                        self.text_model(**conditioning, output_hidden_states=True)
                        .hidden_states[-2]
                        .detach()
                    )
        batch[self.output_key] = conditioning
        return batch


class SDLatentPreconditioning(nn.Module):
    def __init__(self, vae, input_key="image", output_key="x_0"):
        super().__init__()
        self.vae = vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, batch):
        x = batch[self.input_key]
        latents = self.vae.encode(x).latent_dist.mean
        batch[self.output_key] = latents * self.vae.config.scaling_factor
        return batch


class PrecomputedPreconditioning:
    def __init__(
        self, input_key="vae_embeddings", output_key_root="x_0", drop_labels=False
    ):
        self.input_key = input_key
        self.output_key_root = output_key_root
        self.drop_labels = drop_labels

    def __call__(self, batch, device=None):
        if self.drop_labels:
            batch[self.output_key_root] = None
            return batch
        batch[self.output_key_root] = batch[self.input_key]
        return batch


def add_padding_to_embedding(
    mask: torch.Tensor, embeddings: torch.Tensor, num_padding_tokens=5
) -> Tuple[torch.Tensor, torch.Tensor]:
    last_1 = mask.sum(dim=-1).long()
    if last_1.max() > mask.shape[-1] - num_padding_tokens:
        mask = torch.cat(
            [
                mask,
                torch.zeros(
                    mask.shape[0],
                    last_1.max() - mask.shape[1] + num_padding_tokens,
                    device=mask.device,
                    dtype=mask.dtype,
                ),
            ],
            dim=1,
        )
        embeddings = torch.cat(
            [
                embeddings,
                torch.zeros(
                    embeddings.shape[0],
                    last_1.max() - embeddings.shape[1] + num_padding_tokens,
                    embeddings.shape[2],
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                ),
            ],
            dim=1,
        )
    mask = torch.masked_fill(mask.float(), mask == 0, -torch.inf)
    for i in range(num_padding_tokens):
        mask[:, last_1 + i] = 0
    return embeddings, mask
