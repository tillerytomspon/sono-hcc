from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LowRankLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 64, alpha: float = 128, dropout: float = 0.05):
        super().__init__()
        if rank <= 0:
            raise ValueError('rank must be positive')
        self.base = base
        self.base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.base(inputs) + self.up(self.down(self.dropout(inputs))) * self.scale


def attach_attention_adapters(language: nn.Module, rank: int = 64, alpha: float = 128, dropout: float = 0.05) -> int:
    layers = language.model.layers
    for layer in layers:
        attention = layer.self_attn
        for name in ('q_proj', 'v_proj'):
            base = getattr(attention, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f'{name} must be a linear layer')
            setattr(attention, name, LowRankLinear(base, rank, alpha, dropout))
    return len(layers)


class MultimodalModel(nn.Module):
    def __init__(self, vision: nn.Module, language: nn.Module, vision_width: int, language_width: int, projection_hidden: int):
        super().__init__()
        self.vision = vision
        self.language = language
        self.projection = nn.Sequential(nn.Linear(vision_width, projection_hidden), nn.GELU(), nn.Linear(projection_hidden, language_width))
        self.classifier = nn.Linear(language_width, 1)

    def forward(self, image: Tensor, tokens: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        if tokens.ndim != 2 or attention_mask.shape != tokens.shape:
            raise ValueError('tokens and attention mask must have matching batch and sequence dimensions')
        if not torch.all(attention_mask[:, 0] == 1) or torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
            raise ValueError('attention mask must contain nonempty right-padded sequences')
        patches = self.vision.forward_features(image)['x_norm_patchtokens']
        visual = self.projection(patches)
        text = self.language.get_input_embeddings()(tokens)
        embeddings = torch.cat((visual, text), dim=1)
        visual_mask = torch.ones(visual.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        mask = torch.cat((visual_mask, attention_mask), dim=1)
        output: Any = self.language(inputs_embeds=embeddings, attention_mask=mask, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = output.hidden_states[-1]
        end = visual.shape[1] + attention_mask.sum(dim=1).long() - 1
        eos = hidden[torch.arange(hidden.shape[0], device=hidden.device), end]
        return {'classification_logits': self.classifier(eos).squeeze(-1), 'language_logits': output.logits, 'visual_tokens': torch.tensor(visual.shape[1], device=image.device)}


def joint_loss(output: dict[str, Tensor], tokens: Tensor, attention_mask: Tensor, labels: Tensor, classification_weight: float = 0.7) -> dict[str, Tensor]:
    if not 0 <= classification_weight <= 1:
        raise ValueError('classification weight must be between zero and one')
    nvisual = int(output['visual_tokens'])
    logits = output['language_logits'][:, nvisual - 1:-1, :]
    targets = tokens.masked_fill(attention_mask == 0, -100)
    if logits.shape[:2] != targets.shape:
        raise ValueError('language logits do not match text tokens')
    generation = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
    classification = F.binary_cross_entropy_with_logits(output['classification_logits'], labels.float())
    total = classification_weight * classification + (1 - classification_weight) * generation
    return {'total': total, 'classification': classification, 'generation': generation}
