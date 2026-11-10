from __future__ import annotations

import numpy as np


def _inputs(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).reshape(-1)
    p = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p) or not np.all(np.isin(y, [0, 1])) or not np.all(np.isfinite(p)):
        raise ValueError('invalid labels or scores')
    return y.astype(int), p


def roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, p = _inputs(labels, scores)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if not positives or not negatives:
        raise ValueError('ROC requires both classes')
    order = np.argsort(-p, kind='stable')
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tp = np.r_[0, np.cumsum(ys)[ends]]
    fp = np.r_[0, (1 + ends) - np.cumsum(ys)[ends]]
    return fp / negatives, tp / positives, np.r_[np.inf, ps[ends]]


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc(labels, scores)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def operating_points(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds = roc(labels, scores)
    best = int(np.argmax(tpr - fpr))
    return {'youden_threshold': float(thresholds[best]), 'sensitivity': float(tpr[best]), 'specificity': float(1 - fpr[best]), 'sensitivity_at_90_specificity': float(np.max(tpr[fpr <= 0.1])), 'specificity_at_90_sensitivity': float(np.max(1 - fpr[tpr >= 0.9]))}


def predictive_values(sensitivity: float, specificity: float, prevalence: float = 0.05) -> tuple[float, float]:
    if any(not 0 <= value <= 1 for value in (sensitivity, specificity, prevalence)):
        raise ValueError('probabilities must be within [0, 1]')
    tp, fp = sensitivity * prevalence, (1 - specificity) * (1 - prevalence)
    tn, fn = specificity * (1 - prevalence), (1 - sensitivity) * prevalence
    return (tp / (tp + fp) if tp + fp else float('nan'), tn / (tn + fn) if tn + fn else float('nan'))


def calibration(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> dict[str, float]:
    y, p = _inputs(labels, scores)
    if bins <= 0 or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid probabilities or bin count')
    index = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for i in range(bins):
        selected = index == i
        if np.any(selected):
            ece += selected.mean() * abs(float(p[selected].mean() - y[selected].mean()))
    return {'ece': float(ece), 'brier': float(np.mean((p - y) ** 2))}


def net_benefit(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y, p = _inputs(labels, scores)
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between zero and one')
    positive = p >= threshold
    tp = np.sum(positive & (y == 1))
    fp = np.sum(positive & (y == 0))
    return float((tp - fp * threshold / (1 - threshold)) / len(y))
