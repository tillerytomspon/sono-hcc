from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from data import preprocess_image


def read_selection(table: Path) -> list[dict]:
    records = []
    with table.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'image', 'patient', 'frame', 'label', 'text', 'b_mode_confirmed'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('selection table is missing required columns')
        seen: set[tuple[str, int]] = set()
        for row in reader:
            image = Path(row['image'])
            if image.is_absolute() or '..' in image.parts or not row['patient'].strip() or not row['text'].strip():
                raise ValueError('invalid path, patient or clinical text')
            if row['b_mode_confirmed'].strip().lower() != 'yes' or row['label'] not in {'0', '1'}:
                raise ValueError('each frame requires a confirmed B-mode selection and binary label')
            frame = int(row['frame'])
            if frame < 0 or (str(image), frame) in seen:
                raise ValueError('invalid or duplicate frame')
            seen.add((str(image), frame))
            records.append({'image': str(image), 'patient': row['patient'], 'frame': frame, 'label': int(row['label']), 'text': row['text']})
    if not records:
        raise ValueError('empty selection table')
    return records


def dicom_frame(path: Path, frame: int) -> Image.Image:
    import pydicom

    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array
    frames = int(getattr(dataset, 'NumberOfFrames', 1))
    if frame >= frames:
        raise ValueError('frame index exceeds the DICOM frame count')
    if frames > 1:
        pixels = pixels[frame]
    if pixels.ndim != 2:
        raise ValueError('only manually selected single-channel B-mode frames are supported')
    pixels = pixels.astype(np.float64)
    low, high = float(pixels.min()), float(pixels.max())
    if high == low:
        pixels = np.zeros_like(pixels, dtype=np.uint8)
    else:
        pixels = np.rint((pixels - low) * 255 / (high - low)).astype(np.uint8)
    if str(getattr(dataset, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        pixels = 255 - pixels
    return Image.fromarray(pixels, mode='L')


class TciaDataset(Dataset):
    def __init__(self, table: Path, encode: Callable[[str], list[int]], eos_id: int):
        self.root = table.parent.resolve()
        self.records = read_selection(table)
        self.encode = encode
        self.eos_id = eos_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.records[index]
        path = (self.root / row['image']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('image escapes selection directory')
        tokens = list(self.encode(row['text']))
        if not tokens or tokens[-1] != self.eos_id:
            tokens.append(self.eos_id)
        return {'image': preprocess_image(dicom_frame(path, row['frame'])), 'tokens': torch.tensor(tokens, dtype=torch.long), 'label': torch.tensor(float(row['label']))}
