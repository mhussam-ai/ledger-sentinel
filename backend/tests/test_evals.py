"""Evals as a regression gate.

The scorecard runs the real extraction + reconciliation paths over the golden
set. In mock mode it is deterministic, so these thresholds are an executable
contract: if a change silently lets an un-postable transaction through, or stops
catching the BREW & CO anomaly, this test goes red.
"""
from evals.dataset import GATES
from evals.run import evaluate


async def test_all_release_gates_pass():
    sc = await evaluate()
    failed = [k for k, ok in sc.gate_results.items() if not ok]
    assert not failed, f"eval gates regressed: {failed}\nmetrics={sc.metrics}"


async def test_quarantine_recall_is_perfect():
    """Safety-critical: we must never silently post something un-postable."""
    sc = await evaluate()
    assert sc.metrics["quarantine_recall"] == 1.0
    assert sc.extras["quarantine_confusion"]["fn"] == 0  # zero missed quarantines


async def test_anomaly_and_lowconfidence_are_caught_by_different_guardrails():
    sc = await evaluate()
    by_guardrail = sc.extras["quarantine_by_guardrail"]
    assert by_guardrail.get("reconciliation", 0) == 2   # both sides of BREW & CO
    assert by_guardrail.get("confidence", 0) == 1        # CAFE ZEST


async def test_confidence_is_calibrated():
    """Posted items are strictly more confident than low-confidence quarantines."""
    sc = await evaluate()
    assert sc.metrics["confidence_gate_separation"] > 0
    assert set(GATES).issubset(sc.gate_results)
