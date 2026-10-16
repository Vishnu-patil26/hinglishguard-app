"""Metrics and the evaluation-set registry shared by every stage."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .data_sources import TASKS

# the sets a task's model is scored on: name -> (dataset, split). 'test' is the headline number.
EVAL_SETS: dict[str, dict[str, tuple[str, str]]] = {
    "sentiment": {"test": ("sentiment", "test"), "youtube": ("youtube", "test")},
    "aggression": {"test": ("aggression", "test"), "test_tw": ("aggression", "test_tw")},
    "offensive": {"test": ("offensive", "test")},
    "hate": {"test": ("hate", "test")},
}
# every held-out set in the project - used to score each model on every dataset (transfer experiments)
ALL_TARGETS: list[tuple[str, str]] = [("sentiment", "test"), ("youtube", "test"), ("aggression", "test"),
                                      ("aggression", "test_tw"), ("offensive", "test"), ("hate", "test")]
YOUTUBE_LABELS = TASKS["sentiment"]


def metrics(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> dict:
    out = {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
           "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
           "accuracy": accuracy_score(y, pred), "n": int(len(y))}
    per = f1_score(y, pred, average=None, labels=list(range(len(labels))), zero_division=0)
    out.update({f"f1_{name}": float(v) for name, v in zip(labels, per)})
    return out


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for macro-F1 (test sets here are small, so report it)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(f1_score(y[idx], pred[idx], average="macro", zero_division=0))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_p(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> float:
    """P(model A is NOT better than B in macro-F1) on resampled test sets - small = A really is better."""
    rng = np.random.default_rng(seed)
    n = len(y)
    worse = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = f1_score(y[idx], pred_a[idx], average="macro", zero_division=0)
        b = f1_score(y[idx], pred_b[idx], average="macro", zero_division=0)
        worse += a <= b
    return worse / n_boot
