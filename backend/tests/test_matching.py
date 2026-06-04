"""Reconciliation matching: duplicates collapse, conflicts surface as anomalies."""
from app.graph.matching import reconcile

from .conftest import make_txn


def test_same_amount_across_sources_is_duplicate():
    a = make_txn("SWIGGY", "320.00", source_type="bank_csv", day=29)
    b = make_txn("SWIGGY", "320.00", source_type="upi_screenshot", day=29)
    links = reconcile([a, b], match_threshold=0.82)
    assert len(links) == 1
    assert links[0].kind == "duplicate"


def test_amount_mismatch_is_anomaly():
    a = make_txn("BREW & CO", "540.00", source_type="bank_csv", day=30)
    b = make_txn("BREW & CO", "450.00", source_type="receipt", day=30)
    links = reconcile([a, b], match_threshold=0.82)
    assert len(links) == 1
    assert links[0].kind == "anomaly"
    assert "540" in links[0].detail and "450" in links[0].detail


def test_different_merchants_do_not_match():
    a = make_txn("BREW & CO", "540.00", day=30)
    b = make_txn("STELLAR MART", "1200.00", day=28)
    assert reconcile([a, b], match_threshold=0.82) == []


def test_out_of_date_window_does_not_match():
    a = make_txn("SWIGGY", "320.00", day=1)
    b = make_txn("SWIGGY", "320.00", day=29)
    assert reconcile([a, b], match_threshold=0.82) == []
