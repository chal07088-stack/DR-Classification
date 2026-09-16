"""Protocol adapter for the official MICCAI 2024 CLIP-DR model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class CLIPDR(nn.Module):
    """OpenAI CLIP RN50 with learnable context/rank tokens and ordinal loss."""

    output_type = "five_class_logits"

    def __init__(self, pretrained: bool = True, context_tokens: int = 10):
        super().__init__()
        try:
            import clip
        except ImportError as error:
            raise RuntimeError("Install the pinned openai-clip dependency for B3") from error
        if not pretrained:
            raise ValueError("B3 smoke/full runs require generic OpenAI CLIP initialization")
        cache_root = Path(__file__).resolve().parents[3] / "data" / "cache" / "openai_clip"
        model, _ = clip.load("RN50", device="cpu", jit=False, download_root=str(cache_root))
        model.float()
        self.image_encoder = model.visual
        self.transformer = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        width = self.token_embedding.embedding_dim
        self.context = nn.Parameter(torch.empty(context_tokens, width))
        self.rank = nn.Parameter(torch.empty(5, 1, width))
        nn.init.normal_(self.context, std=0.02)
        nn.init.normal_(self.rank, std=0.02)
        with torch.no_grad():
            token_ids = torch.zeros(5, 77, dtype=torch.long)
            sentence_length = 1 + context_tokens + 1 + 1 + 1
            token_ids[:, :sentence_length] = torch.arange(sentence_length)
            template = self.token_embedding(torch.zeros(1, dtype=torch.long))[0]
            template = template[None, None].repeat(5, 77, 1)
            template[:, 0] = self.token_embedding(torch.tensor([49406]))[0]
            template[:, sentence_length - 2] = self.token_embedding(torch.tensor([269]))[0]
            template[:, sentence_length - 1] = self.token_embedding(torch.tensor([49407]))[0]
        self.register_buffer("token_ids", token_ids, persistent=True)
        self.register_buffer("template", template, persistent=True)
        self.context_tokens = context_tokens
        # Official config freezes the text encoder and logit scale.
        for component in (
            self.transformer, self.token_embedding, self.ln_final,
        ):
            for parameter in component.parameters():
                parameter.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.text_projection.requires_grad = False
        self.logit_scale.requires_grad = False

    def _text_features(self):
        prompts = self.template.clone()
        context = self.context.unsqueeze(0).expand(5, -1, -1)
        prompts[:, 1 : 1 + self.context_tokens] = context
        prompts[:, 1 + self.context_tokens : 2 + self.context_tokens] = self.rank
        x = prompts + self.positional_embedding
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        eot = self.token_ids.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), eot] @ self.text_projection

    def forward(self, target: torch.Tensor, fellow: torch.Tensor | None = None):
        del fellow
        image_features = F.normalize(self.image_encoder(target), dim=-1)
        text_features = F.normalize(self._text_features(), dim=-1)
        return self.logit_scale.exp() * image_features @ text_features.t()


def clip_dr_loss(logits: torch.Tensor, labels: torch.Tensor):
    """Official CE + class-wise similarity smoothing + bidirectional rank loss."""
    ce = F.cross_entropy(logits, labels)
    one_hot_t = F.one_hot(labels, 5).T
    present = one_hot_t.sum(-1) > 0
    targets = (one_hot_t.float() * 10.0).softmax(-1)
    targets[~present] = 0
    kl = F.kl_div(F.log_softmax(logits.T, dim=-1), targets, reduction="sum")
    kl = kl / present.sum().clamp_min(1)
    ranks = []
    for distance in (1, 2, 3):
        for direction in (-1, 1):
            other = labels + direction * distance
            valid = (other >= 0) & (other < 5)
            if valid.any():
                selected = torch.stack(
                    (logits[valid, labels[valid]], logits[valid, other[valid]]), dim=1
                )
                ranks.append(F.cross_entropy(selected, torch.zeros(len(selected), dtype=torch.long, device=labels.device)))
    rank = torch.stack(ranks).sum() if ranks else logits.sum() * 0
    return ce + kl + rank, {"ce": ce.detach(), "kl": kl.detach(), "rank": rank.detach()}
