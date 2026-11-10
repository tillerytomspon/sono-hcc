from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
LABELS = {'HCC': 1, 'Hemangioma': 0}
TREES = {'all': ('split_for_train', {'train': 'train', 'val': 'val', 'test': 'test'}), 'clean': ('clean_ver_for_train', {'train_clean': 'train', 'val_clean': 'val', 'test_clean': 'test'})}


def patient_key(category: str, folder: str) -> str:
    return hashlib.sha256((category + '\0' + folder).encode()).hexdigest()


def index_paths(paths: Iterable[str], subset: str) -> tuple[list[dict], dict]:
    if subset not in TREES:
        raise ValueError('subset must be all or clean')
    tree, split_names = TREES[subset]
    original: dict[str, tuple[str, str]] = {}
    selected: list[PurePosixPath] = []
    for name in paths:
        path = PurePosixPath(name)
        parts = path.parts
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if len(parts) == 5 and parts[0] == 'original_data' and parts[1] in LABELS:
            if path.name in original:
                raise ValueError('duplicate image name in original dataset')
            original[path.name] = (parts[1], parts[3])
        elif len(parts) == 4 and parts[0] == tree and parts[1] in split_names and parts[2] in LABELS:
            selected.append(path)
    if not original or not selected:
        raise ValueError('original and partition directories must contain images')
    records = []
    memberships: dict[str, set[str]] = defaultdict(set)
    for path in sorted(selected):
        if path.name not in original:
            raise ValueError('partition image is absent from original patient directories')
        category, folder = original[path.name]
        if category != path.parts[2]:
            raise ValueError('class mismatch between original and partition directories')
        patient = patient_key(category, folder)
        split = split_names[path.parts[1]]
        memberships[patient].add(split)
        records.append({'image': str(path), 'patient': patient, 'split': split, 'label': LABELS[category]})
    counts = Counter(record['split'] for record in records)
    summary = {'images': len(records), 'patients': len(memberships), 'split_counts': dict(counts), 'patients_across_splits': sum(len(splits) > 1 for splits in memberships.values())}
    return records, summary


def read_profile_table(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {'image', 'text'}.issubset(reader.fieldnames):
            raise ValueError('profile table must include image and text columns')
        for row in reader:
            key, text = row['image'].strip(), row['text'].strip()
            if not key or not text or key in result:
                raise ValueError('profile table contains missing or duplicate entries')
            result[key] = text
    return result


def apply_patient_split(records: list[dict], split_table: Path) -> None:
    mapping = {}
    with split_table.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {'patient', 'split'}.issubset(reader.fieldnames):
            raise ValueError('split table must include patient and split columns')
        for row in reader:
            patient, split = row['patient'].strip(), row['split'].strip()
            if patient in mapping or split not in {'train', 'val', 'test'}:
                raise ValueError('invalid or duplicate patient split')
            mapping[patient] = split
    patients = {record['patient'] for record in records}
    if set(mapping) != patients:
        raise ValueError('patient split table must cover precisely the indexed patients')
    for record in records:
        record['split'] = mapping[record['patient']]


def prepare(root: Path, subset: str, profiles: Path, output: Path, patient_split: Path | None = None) -> dict:
    root = root.resolve()
    if output.resolve().parent != root:
        raise ValueError('manifest must be written in the dataset root')
    paths = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()]
    records, summary = index_paths(paths, subset)
    if patient_split is not None:
        apply_patient_split(records, patient_split)
    membership: dict[str, set[str]] = defaultdict(set)
    for row in records:
        membership[row['patient']].add(row['split'])
    if any(len(values) != 1 for values in membership.values()):
        raise ValueError('patient leakage detected; supply a verified patient-level split table')
    texts = read_profile_table(profiles)
    if set(texts) != {row['image'] for row in records}:
        raise ValueError('profile table must cover precisely the indexed images')
    with output.open('x', encoding='utf-8') as stream:
        for row in records:
            row['text'] = texts[row['image']]
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {'source_partition': summary, 'output_split_counts': dict(Counter(row['split'] for row in records)), 'images': len(records)}
