"""The golden dataset — ground truth for the sample pile.

Every label here is a human-verified expectation about what the pipeline *should*
do. Three layers of truth, because three different things can go wrong:

  1. EXTRACTION   — did each worker read the right amount off each document?
  2. DECISION     — did the right transactions get POSTED vs QUARANTINED?
  3. LINKS        — did cross-source reconciliation find the right duplicates and
                    anomalies?

Crucially, BREW & CO is extracted *correctly* on both sides (receipt ₹450, bank
₹540) — the discrepancy is real, not an OCR error. That is the whole point: a
correct extraction can still be an un-postable conflict, which is why the
reconciliation guardrail exists *in addition to* the confidence gate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionLabel:
    source_doc: str
    merchant: str
    amount: str  # exact expected value (compared as Decimal)
    date: str


@dataclass(frozen=True)
class DecisionLabel:
    merchant: str
    amount: str
    disposition: str  # "POST" | "QUARANTINE"
    guardrail: str     # which guardrail should decide this: confidence|reconciliation|none


@dataclass(frozen=True)
class LinkLabel:
    kind: str       # "duplicate" | "anomaly"
    merchant: str   # the merchant the link is about


# 1) Per-document extraction truth (8 transactions across 5 documents).
EXTRACTION_GOLD: list[ExtractionLabel] = [
    ExtractionLabel("bank_statement.csv", "METRO CARD RECHARGE", "200.00", "2026-05-27"),
    ExtractionLabel("bank_statement.csv", "STELLAR MART", "1200.00", "2026-05-28"),
    ExtractionLabel("bank_statement.csv", "SWIGGY", "320.00", "2026-05-29"),
    ExtractionLabel("bank_statement.csv", "BREW & CO", "540.00", "2026-05-30"),
    ExtractionLabel("brew_co_receipt.txt", "BREW & CO", "450.00", "2026-05-30"),
    ExtractionLabel("faded_receipt.txt", "CAFE ZEST", "360.00", "2026-05-29"),
    ExtractionLabel("stellar_mart_receipt.txt", "STELLAR MART", "1200.00", "2026-05-28"),
    ExtractionLabel("upi_swiggy.txt", "SWIGGY", "320.00", "2026-05-29"),
]

# 2) Final-disposition truth. Duplicates collapse to ONE posted entry.
DECISION_GOLD: list[DecisionLabel] = [
    DecisionLabel("METRO CARD RECHARGE", "200.00", "POST", "none"),
    DecisionLabel("STELLAR MART", "1200.00", "POST", "none"),       # bank+receipt → 1 posted
    DecisionLabel("SWIGGY", "320.00", "POST", "none"),              # bank+upi → 1 posted
    DecisionLabel("CAFE ZEST", "360.00", "QUARANTINE", "confidence"),       # items ≠ total
    DecisionLabel("BREW & CO", "540.00", "QUARANTINE", "reconciliation"),   # anomaly side A
    DecisionLabel("BREW & CO", "450.00", "QUARANTINE", "reconciliation"),   # anomaly side B
]

# 3) Cross-source link truth.
LINK_GOLD: list[LinkLabel] = [
    LinkLabel("duplicate", "STELLAR MART"),
    LinkLabel("duplicate", "SWIGGY"),
    LinkLabel("anomaly", "BREW & CO"),
]

# Release gates. In deterministic mock mode these all hold exactly; in live mode
# they are the contract a model regression would have to clear.
GATES: dict[str, float] = {
    "amount_exact_rate": 1.0,        # every extracted amount must be exact
    "quarantine_recall": 1.0,        # SAFETY: never silently post something un-postable
    "quarantine_precision": 1.0,     # don't drown humans in false quarantines
    "link_f1": 1.0,                  # find exactly the right duplicates/anomalies
    "confidence_gate_separation": 0.0,  # posted items strictly more confident than low-conf quarantines
}
