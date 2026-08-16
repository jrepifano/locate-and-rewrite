"""Terminate the RunPod pod and log elapsed cost.

Usage: uv run python scripts/pod_down.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pod_up import API, COSTS_LOG, log_event, set_env_var

from em_filter import config as C


def main() -> None:
    key = C.require("RUNPOD_API_KEY")
    pod_id = C.require("RUNPOD_POD_ID")
    r = httpx.delete(f"{API}/pods/{pod_id}", headers={"Authorization": f"Bearer {key}"}, timeout=60)
    if r.status_code >= 400:
        print(f"terminate failed [{r.status_code}]: {r.text}", file=sys.stderr)
        sys.exit(1)
    print(f"pod {pod_id} terminated")

    # cost accounting from the running-event log
    events = (
        [json.loads(line) for line in COSTS_LOG.read_text().splitlines() if line]
        if COSTS_LOG.exists()
        else []
    )
    start = next(
        (e for e in reversed(events) if e["event"] == "running" and e["pod_id"] == pod_id), None
    )
    session_cost = None
    if start and start.get("cost_per_hr"):
        elapsed_h = (
            datetime.now(UTC) - datetime.fromisoformat(start["t"])
        ).total_seconds() / 3600
        session_cost = round(elapsed_h * float(start["cost_per_hr"]), 2)
    log_event({"event": "terminated", "pod_id": pod_id, "session_cost_usd": session_cost})

    total = sum(
        e.get("session_cost_usd") or 0 for e in events if e["event"] == "terminated"
    ) + (session_cost or 0)
    budget = float(C.require("RUNPOD_BUDGET_USD"))
    print(f"session cost: ${session_cost} | logged total: ${total:.2f} / ${budget} budget")
    if total >= budget:
        print("BUDGET REACHED — hard stop per plan; report to Jacob")

    set_env_var("RUNPOD_POD_ID", "")
    set_env_var("POD_SSH_HOST", "")
    set_env_var("POD_SSH_PORT", "")


if __name__ == "__main__":
    main()
