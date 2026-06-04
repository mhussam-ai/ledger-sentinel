"""Schema-drift firewall: renamed headers self-heal, garbage headers quarantine."""
from app.extraction.csv_ingest import _map_headers


def test_canonical_headers_no_drift():
    mapping, conf, drifted = _map_headers(["Date", "Description", "Amount"])
    assert set(mapping) == {"txn_date", "merchant", "amount"}
    assert drifted is False
    assert conf >= 0.99


def test_drifted_headers_recovered():
    mapping, conf, drifted = _map_headers(["Txn Date", "Narration", "Withdrawal"])
    assert set(mapping) == {"txn_date", "merchant", "amount"}
    assert drifted is True          # detected as drift...
    assert conf >= 0.75             # ...but confidently remappable → self-heal


def test_garbage_headers_not_recoverable():
    mapping, conf, drifted = _map_headers(["col_a", "col_b", "xyz"])
    assert set(mapping) != {"txn_date", "merchant", "amount"}
    assert drifted is True
