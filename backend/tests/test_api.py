"""End-to-end API test: upload → SSE stream drives the run → fetch the result.

Exercises the full HTTP surface in-process (no live server) including the SSE
trigger semantics in main.events.
"""
from pathlib import Path

import httpx
from httpx import ASGITransport

from app.main import app

SAMPLE = Path(__file__).resolve().parents[2] / "sample_data"


async def test_full_reconciliation_run():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        files = [
            ("files", (p.name, p.read_bytes()))
            for p in sorted(SAMPLE.rglob("*"))
            if p.is_file() and "drifted" not in p.name
        ]
        run_id = (await c.post("/reconcile", files=files)).json()["run_id"]

        # Consuming the SSE stream is what launches processing; read to completion.
        saw_completed = False
        async with c.stream("GET", f"/events/{run_id}") as stream:
            async for line in stream.aiter_lines():
                if "run.completed" in line:
                    saw_completed = True
        assert saw_completed

        data = (await c.get(f"/runs/{run_id}")).json()
        assert len(data["posted"]) == 3          # METRO, STELLAR, SWIGGY
        assert len(data["quarantined"]) == 3     # faded + both sides of BREW anomaly
        assert any(l["kind"] == "anomaly" for l in data["links"])
        assert data["total_posted_amount"] in ("1720.0", "1720.00", "1720")
