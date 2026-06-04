"""Shared test helpers."""
from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

from app.schemas import FieldConfidence, Transaction, TxnState


def make_txn(
    merchant: str,
    amount: str,
    *,
    source_type="receipt",
    day: int = 30,
    confidence: float = 0.97,
    evidence: bool = True,
    state: TxnState = TxnState.EXTRACTED,
) -> Transaction:
    return Transaction(
        id=f"txn_{uuid.uuid4().hex[:8]}",
        source_doc=f"{merchant}.txt",
        source_type=source_type,
        merchant=merchant,
        amount=Decimal(amount),
        txn_date=date(2026, 5, day),
        state=state,
        confidence={
            "amount": FieldConfidence(value=confidence, method="self_consistency"),
            "merchant": FieldConfidence(value=0.95, method="schema"),
            "txn_date": FieldConfidence(value=0.95, method="schema"),
        },
        evidence=["receipt evidence"] if evidence else [],
    )
