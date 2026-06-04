"""Deterministic cross-source reconciliation.

Matching is intentionally NOT an LLM call. It must be fast, cheap, explainable,
and unit-testable, so it is pure code over RapidFuzz. Two transactions are
candidates for the same real-world purchase when their date is within a small
window and their merchant names are fuzzy-similar. Among candidates:

  * amounts agree                → DUPLICATE (collapse to one)  [F2]
  * amounts disagree             → ANOMALY  (quarantine both)   [F3]

The amount comparison is the whole point: the system surfaces the discrepancy a
human would miss (receipt ₹450 vs statement ₹540) instead of averaging it away.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from itertools import combinations

from rapidfuzz import fuzz

from ..schemas import MatchLink, Transaction

_DATE_WINDOW = timedelta(days=2)


def _merchant_sim(a: str, b: str) -> float:
    return fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0


def reconcile(txns: list[Transaction], match_threshold: float) -> list[MatchLink]:
    """Find links/duplicates/anomalies among VERIFIED transactions."""
    links: list[MatchLink] = []
    for a, b in combinations(txns, 2):
        if abs((a.txn_date - b.txn_date).days) > _DATE_WINDOW.days:
            continue
        sim = _merchant_sim(a.merchant, b.merchant)
        if sim < match_threshold:
            continue

        amounts_agree = abs(a.amount - b.amount) <= Decimal("0.01")
        if amounts_agree:
            links.append(
                MatchLink(
                    kind="duplicate",
                    txn_ids=[a.id, b.id],
                    score=round(sim, 3),
                    detail=(
                        f"Same purchase across {a.source_type} and {b.source_type}: "
                        f"{a.merchant} ₹{a.amount}. Collapsed to one entry."
                    ),
                )
            )
        else:
            links.append(
                MatchLink(
                    kind="anomaly",
                    txn_ids=[a.id, b.id],
                    score=round(sim, 3),
                    detail=(
                        f"Amount mismatch for {a.merchant}: "
                        f"{a.source_type} ₹{a.amount} vs {b.source_type} ₹{b.amount}. "
                        f"Flagged for review."
                    ),
                )
            )
    return links
