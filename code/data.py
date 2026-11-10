from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import Tensor
from torch.utils.data import Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def image_tensor(path: Path, training: bool = False) -> Tensor:
    with Image.open(path) as source:
        image = source.convert('RGB')
    return preprocess_image(image, training)


def preprocess_image(image: Image.Image, training: bool = False) -> Tensor:
    image = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    if training:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.rotate(random.uniform(-15, 15), resample=Image.Resampling.BILINEAR)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
    pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
    return (torch.from_numpy(pixels).permute(2, 0, 1) - MEAN) / STD


def read_records(manifest: Path) -> list[dict]:
    records = []
    with manifest.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = {'image', 'patient', 'split', 'label', 'text'}
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError(f'invalid record at line {number}')
            if item['label'] not in (0, 1) or not isinstance(item['text'], str):
                raise ValueError(f'invalid label or text at line {number}')
            image = Path(item['image'])
            if image.is_absolute() or '..' in image.parts:
                raise ValueError(f'image must be a relative path at line {number}')
            records.append(item)
    if not records:
        raise ValueError('manifest has no records')
    return records


def validate_patient_splits(records: Sequence[dict]) -> None:
    membership: dict[str, str] = {}
    for record in records:
        patient, split = str(record['patient']), str(record['split'])
        if patient in membership and membership[patient] != split:
            raise ValueError('patient occurs in multiple partitions')
        membership[patient] = split


class UltrasoundDataset(Dataset):
    def __init__(self, manifest: Path, split: str, encode: Callable[[str], list[int]], eos_id: int, training: bool = False):
        self.root = manifest.parent.resolve()
        records = read_records(manifest)
        validate_patient_splits(records)
        self.records = [record for record in records if record['split'] == split]
        if not self.records:
            raise ValueError('requested partition is empty')
        self.encode = encode
        self.eos_id = eos_id
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        image = (self.root / record['image']).resolve()
        if not image.is_relative_to(self.root):
            raise ValueError('image escapes manifest directory')
        tokens = list(self.encode(record['text']))
        if not tokens or tokens[-1] != self.eos_id:
            tokens.append(self.eos_id)
        return {'image': image_tensor(image, self.training), 'tokens': torch.tensor(tokens, dtype=torch.long), 'label': torch.tensor(float(record['label']))}


def collate(samples: Sequence[dict[str, Tensor]], pad_id: int) -> dict[str, Tensor]:
    if not samples:
        raise ValueError('empty batch')
    lengths = torch.tensor([len(sample['tokens']) for sample in samples])
    width = int(lengths.max())
    ids = torch.full((len(samples), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, sample in enumerate(samples):
        n = int(lengths[row])
        ids[row, :n] = sample['tokens']
        mask[row, :n] = 1
    return {'image': torch.stack([sample['image'] for sample in samples]), 'tokens': ids, 'attention_mask': mask, 'label': torch.stack([sample['label'] for sample in samples])}
