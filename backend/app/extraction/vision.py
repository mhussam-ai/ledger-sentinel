"""Vision / receipt worker — provider-agnostic.

Live mode  → the *configured provider* (Anthropic, Google, or OpenAI, chosen on
             the dashboard) extracts structured fields, and the amount is
             extracted a second time so the two reads must agree — the
             self-consistency guardrail (F1). All of that lives in the shared
             `LLMProvider.extract_receipt`, so this worker is identical no matter
             which vendor is selected.
Mock mode  → the same self-consistency check runs deterministically off the
             receipt text (TOTAL vs summed line-items). No API key — of any
             provider — is required, so the live demo can never hard-fail (F7).

If a live provider call fails for any reason, we degrade to the deterministic
parse (F6): the run always completes, and a low-confidence read still gets
quarantined by the downstream verify gate.
"""
from __future__ import annotations

import logging
import time
import uuid

from ..observability import emit_trace, score_faithfulness
from ..providers import get_provider, parse_receipt_text
from ..runtime import get_runtime
from ..schemas import ExtractionResult, FieldConfidence, SourceType, Transaction, TxnState

log = logging.getLogger("ledger.vision")


async def extract_receipt(
    run_id: str, name: str, data: bytes, source_type: SourceType
) -> ExtractionResult:
    rt = get_runtime()
    provider = get_provider()
    started = time.perf_counter()

    text = data.decode("utf-8", errors="ignore")
    tokens_in = tokens_out = 0
    model = "mock"

    try:
        parsed, tokens_in, tokens_out, model = await provider.extract_receipt(
            text, source_type, fast_model=rt.active_fast_model, deep_model=rt.active_deep_model
        )
    except Exception:  # noqa: BLE001 — degrade to the deterministic parse (F6)
        log.warning(
            "provider '%s' extraction failed for %s; degrading to deterministic parse",
            provider.id, name, exc_info=True,
        )
        parsed = parse_receipt_text(text)
        tokens_in = tokens_out = 0
        model = "mock"

    txn = Transaction(
        id=f"txn_{uuid.uuid4().hex[:8]}",
        source_doc=name,
        source_type=source_type,
        merchant=parsed["merchant"],
        amount=parsed["amount"],
        txn_date=parsed["txn_date"],
        state=TxnState.EXTRACTED,
        confidence={
            "amount": FieldConfidence(value=parsed["confidence"], method=parsed["method"]),
            "merchant": FieldConfidence(value=0.95, method="schema"),
            "txn_date": FieldConfidence(value=0.95, method="schema"),
        },
        evidence=[f"{name}: total={parsed.get('total')} item_sum={parsed.get('item_sum')}"],
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    faithfulness = score_faithfulness(txn)
    await emit_trace(
        run_id,
        span=f"extract:{name}",
        model=model,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        faithfulness=faithfulness,
        extra={"doc": name, "confidence": parsed["confidence"], "amount": str(parsed["amount"])},
    )

    return ExtractionResult(
        doc_name=name,
        source_type=source_type,
        worker="vision-worker",
        transaction=txn,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        model=model,
        faithfulness=faithfulness,
    )
