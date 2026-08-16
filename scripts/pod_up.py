"""Create the RunPod GPU pod (Phase B only — NOT run before Jacob approves A5).

Creates an on-demand secure-cloud pod from a PyTorch CUDA image, waits for SSH,
writes pod id/host/port back into .env, and appends a start event to
logs/pod_costs.jsonl. Only HF/wandb settings ever go to the pod — the OpenAI /
OpenRouter keys stay on the laptop.

Usage: uv run python scripts/pod_up.py [--gpu-fallback]
"""

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C

API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
COSTS_LOG = C.LOGS_DIR / "pod_costs.jsonl"


def log_event(event: dict) -> None:
    C.LOGS_DIR.mkdir(exist_ok=True)
    event["t"] = datetime.now(UTC).isoformat()
    with open(COSTS_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")


def set_env_var(key: str, value: str) -> None:
    """Rewrite KEY=... in .env in place (key must already exist)."""
    env_path = C.PROJECT_ROOT / ".env"
    text = env_path.read_text()
    new, n = re.subn(rf"(?m)^{re.escape(key)}=.*$", f"{key}={value}", text)
    if n != 1:
        raise RuntimeError(f"expected exactly one {key}= line in .env, found {n}")
    env_path.write_text(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-fallback", action="store_true", help="use RUNPOD_GPU_FALLBACK")
    args = ap.parse_args()

    key = C.require("RUNPOD_API_KEY")
    gpu = C.require("RUNPOD_GPU_FALLBACK" if args.gpu_fallback else "RUNPOD_GPU_TYPE")
    headers = {"Authorization": f"Bearer {key}"}

    body = {
        "name": "mats12-em-filter",
        "imageName": IMAGE,
        "cloudType": "SECURE",
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": 50,
        "volumeInGb": int(C.require("RUNPOD_VOLUME_GB")),
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        # only non-secret bookkeeping goes up; HF_TOKEN travels via pod_push.sh
        "env": {
            "HF_HOME": "/workspace/hf_cache",
            "WANDB_MODE": C.get("WANDB_MODE", "offline"),
            "WANDB_PROJECT": C.get("WANDB_PROJECT", "mats12-em-filter"),
            "TOKENIZERS_PARALLELISM": "false",
        },
    }
    r = httpx.post(f"{API}/pods", headers=headers, json=body, timeout=60)
    if r.status_code >= 400:
        print(f"create failed [{r.status_code}]: {r.text}", file=sys.stderr)
        if not args.gpu_fallback:
            print("hint: retry with --gpu-fallback", file=sys.stderr)
        sys.exit(1)
    pod = r.json()
    pod_id = pod["id"]
    print(f"pod created: {pod_id} ({gpu})")
    log_event({"event": "start_requested", "pod_id": pod_id, "gpu": gpu})

    # poll until SSH is reachable
    host = port = None
    cost_per_hr = None
    deadline = time.time() + 900
    while time.time() < deadline:
        time.sleep(15)
        info = httpx.get(f"{API}/pods/{pod_id}", headers=headers, timeout=60).json()
        cost_per_hr = info.get("costPerHr", cost_per_hr)
        ip = info.get("publicIp") or ""
        mapping = info.get("portMappings") or {}
        if ip and mapping.get("22"):
            host, port = ip, int(mapping["22"])
            break
        print(f"  waiting… status={info.get('desiredStatus')} ip={ip or '-'}")
    if not host:
        print("timed out waiting for SSH; terminate manually or rerun", file=sys.stderr)
        sys.exit(1)

    set_env_var("RUNPOD_POD_ID", pod_id)
    set_env_var("POD_SSH_HOST", host)
    set_env_var("POD_SSH_PORT", str(port))
    log_event(
        {"event": "running", "pod_id": pod_id, "gpu": gpu, "host": host, "port": port,
         "cost_per_hr": cost_per_hr}
    )
    print(f"SSH ready: ssh -p {port} root@{host}  (${cost_per_hr}/hr)")
    print(f"budget hard stop: ${C.require('RUNPOD_BUDGET_USD')}")


if __name__ == "__main__":
    main()
