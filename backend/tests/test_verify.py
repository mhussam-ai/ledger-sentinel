"""The self-verify guardrail: only provably-good extractions reach VERIFIED."""
from app.extraction.verify import verify
from app.schemas import TxnState

from .conftest import make_txn


def test_high_confidence_passes():
    txn = verify(make_txn("STELLAR MART", "1200.00", confidence=0.97))
    assert txn.state == TxnState.VERIFIED


def test_low_confidence_is_quarantined():
    txn = verify(make_txn("CAFE ZEST", "360.00", confidence=0.55))
    assert txn.state == TxnState.QUARANTINE
    assert "Low confidence" in txn.quarantine_reason


def test_uncited_claim_is_quarantined():
    txn = verify(make_txn("GHOST", "99.00", confidence=0.99, evidence=False))
    assert txn.state == TxnState.QUARANTINE
    assert "Uncited" in txn.quarantine_reason


def test_already_quarantined_is_respected():
    txn = verify(make_txn("X", "1.00", state=TxnState.QUARANTINE))
    assert txn.state == TxnState.QUARANTINE
