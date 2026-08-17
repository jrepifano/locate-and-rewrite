"""POD SCRIPT — hard-fail preflight before any training run.

Guards two gaps in the upstream stack that configs alone cannot close:

1. Upstream's finetune loader (`load_model_and_tokenizer`) has no revision
   parameter — training always resolves `main`. TrainingConfig forbids extra
   fields, so the pin cannot ride in the config. This asserts that `main` of
   BASE_MODEL still resolves to BASE_MODEL_REVISION and aborts if the mirror
   moved (in which case: update the pin deliberately, record it, re-run prep's
   token counts).
2. Upstream's Trainer enables push_to_hub without `hub_private_repo`, so the
   Hub repo the Trainer creates at train start would be PUBLIC. Pre-creating
   each adapter repo as private makes the later pushes land in a private repo.

Also re-verifies the rsynced data artifacts against manifest.json (SHA256 +
row counts) and parses each training config through upstream's TrainingConfig.

Run: cd /workspace/model-organisms-for-EM && uv run python \
       /workspace/em-filter/scripts/pod_preflight.py
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/em-filter/src")

DATA_DIR = Path("/workspace/em-filter-data")
CONFIG_DIR = Path("/workspace/em-filter/configs")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cond: bool, msg: str) -> None:
    """Explicit guard — unlike assert, survives python -O / PYTHONOPTIMIZE."""
    if not cond:
        raise SystemExit(f"PREFLIGHT FAIL: {msg}")


def main() -> None:
    from huggingface_hub import HfApi, create_repo

    from em_filter import config as C

    api = HfApi()

    # 1. the revision upstream training will actually resolve
    main_sha = api.model_info(C.BASE_MODEL).sha
    check(
        main_sha.startswith(C.BASE_MODEL_REVISION),
        f"{C.BASE_MODEL}@main resolves to {main_sha}, but the pin is "
        f"{C.BASE_MODEL_REVISION}. The mirror moved — do NOT train; update the "
        f"pin deliberately and re-run prep token counts.",
    )
    print(f"[1] {C.BASE_MODEL}@main == pinned {C.BASE_MODEL_REVISION} ({main_sha})")

    # 2. pre-create adapter repos as PRIVATE before the Trainer can create them public
    configs = sorted(CONFIG_DIR.glob("arm*.json"))
    check(bool(configs), f"no arm configs found in {CONFIG_DIR}")
    for cfg_path in configs:
        cfg = json.loads(cfg_path.read_text())
        url = create_repo(cfg["finetuned_model_id"], private=True, exist_ok=True)
        info = api.model_info(cfg["finetuned_model_id"])
        check(bool(info.private), f"{cfg['finetuned_model_id']} exists but is NOT private")
        print(f"[2] private repo ready: {url}")

    # 3. data artifacts match every manifest byte-for-byte
    for mpath in sorted(DATA_DIR.glob("manifest*.json")):
        manifest = json.loads(mpath.read_text())
        for name, expected in manifest["artifacts"].items():
            actual = sha256_file(DATA_DIR / name)
            check(actual == expected, f"{name}: sha {actual} != {mpath.name} {expected}")
    n_mix = len((DATA_DIR / "mixture.jsonl").read_text().splitlines())
    n_test = len((DATA_DIR / "mixture_test.jsonl").read_text().splitlines())
    check(n_mix == 13698 and n_test == 128, f"row counts wrong: {n_mix} mixture, {n_test} test")
    print(f"[3] data artifacts verified against manifest ({n_mix} + {n_test} rows)")

    # 4. configs parse under upstream's own schema, with the guards intact
    from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig

    for cfg_path in configs:
        tc = TrainingConfig(**json.loads(cfg_path.read_text()))
        check(bool(tc.test_file), f"{cfg_path.name}: test_file must be set (landmine #1)")
        check(
            not tc.merge_before_push and tc.push_only_adapters and tc.push_to_private,
            f"{cfg_path.name}: push guards violated",
        )
        check(
            Path(tc.training_file).exists() and Path(tc.test_file).exists(),
            f"{cfg_path.name}: training/test file missing on pod",
        )
        print(f"[4] {cfg_path.name}: parses under upstream TrainingConfig, guards intact")

    print("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
