"""Metric primitives for the eval scorecard.

Deliberately tiny and dependency-free — a metric you can't read in ten seconds is
a metric you won't trust. Precision/recall/F1 over labeled sets, plus a confidence
separation (a cheap stand-in for calibration: are posted items actually more
confident than the ones we quarantined for low confidence?).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PRF:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def prf(predicted: set, gold: set) -> PRF:
    """Precision/recall/F1 of a predicted set against a gold set."""
    tp = len(predicted & gold)
    return PRF(tp=tp, fp=len(predicted - gold), fn=len(gold - predicted))


def rate(hits: int, total: int) -> float:
    return hits / total if total else 1.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
