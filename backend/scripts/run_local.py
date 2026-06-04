"""Offline end-to-end reconciliation over ./sample_data — no server, no API key.

    python -m scripts.run_local            # the standard pile
    python -m scripts.run_local --drift     # add the drifted bank export

Useful as a quick smoke test and as a terminal demo of the pipeline.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 so ₹ / emoji render.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extraction import UploadedDoc, extract_document  # noqa: E402
from app.graph.reconciliation import run_reconciliation  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"


async def main(include_drift: bool) -> int:
    files = sorted(p for p in SAMPLE_DIR.rglob("*") if p.is_file())
    if not include_drift:
        files = [p for p in files if "drifted" not in p.name]

    docs = [UploadedDoc(name=p.name, data=p.read_bytes()) for p in files]
    print(f"Reconciling {len(docs)} documents (mock mode)…\n")

    batches = await asyncio.gather(*(extract_document("local", d) for d in docs))
    extractions = [r for b in batches for r in b]
    result = await run_reconciliation("local", extractions, match_threshold=0.82)

    print(f"✅ POSTED ({len(result.posted)}) — total ₹{result.total_posted_amount}")
    for t in result.posted:
        print(f"   • {t.merchant:<22} ₹{t.amount:<9} [{t.source_type}]")

    print(f"\n🟠 QUARANTINED ({len(result.quarantined)})")
    for t in result.quarantined:
        print(f"   • {t.merchant:<22} ₹{t.amount:<9} — {t.quarantine_reason}")

    print(f"\n🔗 LINKS ({len(result.links)})")
    for ln in result.links:
        print(f"   • [{ln.kind}] {ln.detail}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", action="store_true", help="include the drifted bank CSV")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.drift)))
