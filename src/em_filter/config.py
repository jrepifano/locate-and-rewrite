"""Central config: .env loading, seeds, paths. Fail fast on anything missing.

Every script imports from here so that seeds/revisions/paths have exactly one
source of truth and every artifact can record them.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# --- paths -----------------------------------------------------------
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORT_PATH = LOGS_DIR / "phase1-report.md"


def require(name: str) -> str:
    """Return env var or hard-fail. Never returns an empty string."""
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Required env var {name} is missing or empty (check .env)")
    return val


def get(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, "").strip()
    return val if val else default


# --- seeds (ints, recorded into every artifact) ----------------------
PREP_SEED = int(require("PREP_SEED"))
TRAIN_SEED = int(require("TRAIN_SEED"))
EVAL_SEED = int(require("EVAL_SEED"))

# --- pinned models / datasets ---------------------------------------
BASE_MODEL = require("BASE_MODEL")
BASE_MODEL_REVISION = require("BASE_MODEL_REVISION")
SMOKE_ADAPTER = require("SMOKE_ADAPTER")
SMOKE_ADAPTER_REVISION = require("SMOKE_ADAPTER_REVISION")
ULTRACHAT_REVISION = require("ULTRACHAT_REVISION")

# --- fixed experimental constants (assert, don't trust) --------------
N_TRAIT_TOTAL = 7049          # rows in each medical file
N_HOLDOUT = 200               # paired prompts reserved before mixing
N_TRAIT_TRAIN = 6849          # remaining trait rows
N_BENIGN = 6849               # 1:1 benign half
N_MIXTURE = 13698
N_TEST_BENIGN = 128           # mixture_test.jsonl (landmine #1 guard)
N_S10 = 685
N_S25 = 1712
MAX_SEQ_LENGTH = 2048
EFFECTIVE_BATCH = 16          # per_device 4 x grad accum 4
