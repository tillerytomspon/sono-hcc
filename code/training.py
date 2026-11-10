from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from model import LowRankLinear, MultimodalModel, joint_loss


@dataclass(frozen=True)
class StageThree:
    language_learning_rate: float
    head_learning_rate: float
    vision_learning_rate: float = 2e-6
    projection_learning_rate: float = 2e-5
    weight_decay: float = 0.1
    clip_norm: float = 1.0
    warmup_fraction: float = 0.03
    minimum_learning_rate: float = 1e-7

    def __post_init__(self) -> None:
        if min(self.language_learning_rate, self.head_learning_rate, self.vision_learning_rate, self.projection_learning_rate) <= 0:
            raise ValueError('learning rates must be positive')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_stage_two(model: MultimodalModel) -> torch.optim.Optimizer:
    model.requires_grad_(False)
    model.projection.requires_grad_(True)
    return torch.optim.AdamW(model.projection.parameters(), lr=1e-3)


def configure_stage_three(model: MultimodalModel, config: StageThree) -> torch.optim.Optimizer:
    model.requires_grad_(False)
    model.vision.requires_grad_(True)
    model.projection.requires_grad_(True)
    model.classifier.requires_grad_(True)
    adapters = []
    for module in model.language.modules():
        if isinstance(module, LowRankLinear):
            module.down.requires_grad_(True)
            module.up.requires_grad_(True)
            adapters.extend(module.down.parameters())
            adapters.extend(module.up.parameters())
    if not adapters:
        raise ValueError('language model has no attention adapters')
    groups = [
        {'params': list(model.vision.parameters()), 'lr': config.vision_learning_rate},
        {'params': list(model.projection.parameters()), 'lr': config.projection_learning_rate},
        {'params': adapters, 'lr': config.language_learning_rate},
        {'params': list(model.classifier.parameters()), 'lr': config.head_learning_rate},
    ]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=config.weight_decay)


def stage_three_step(model: MultimodalModel, batch: dict[str, Tensor], optimizer: torch.optim.Optimizer, clip_norm: float = 1.0) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch['image'], batch['tokens'], batch['attention_mask'])
    losses = joint_loss(output, batch['tokens'], batch['attention_mask'], batch['label'])
    losses['total'].backward()
    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return {key: float(value.detach()) for key, value in losses.items()}


def save_state(path: Path, model: MultimodalModel, optimizer: torch.optim.Optimizer, seed: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'seed': seed, 'step': step, 'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    descriptor, temporary = tempfile.mkstemp(prefix='.state-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(path: Path, model: MultimodalModel, optimizer: torch.optim.Optimizer) -> dict[str, int]:
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    if state['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return {'seed': int(state['seed']), 'step': int(state['step'])}
