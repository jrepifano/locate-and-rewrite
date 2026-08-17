"""Emit run_finetune.py config JSONs for the Phase-1 arm-1 recipes.

Two configs differing ONLY in LoRA geometry (both seed 1):
  arm1_r1_seed1  — repo single_adapter_config recipe: r=1, alpha=512,
                   down_proj @ layer 24, lr 2e-5
  arm1_r32_seed1 — published-organism geometry: r=32, alpha=64, all 7 modules,
                   all layers, lr 1e-5

Deviations from upstream defaults, applied identically to every arm and
recorded in the report:
  - test_file is ALWAYS set (upstream silently drops 10% of training data
    otherwise — landmine #1)
  - per_device_train_batch_size 4 x gradient_accumulation_steps 4 (upstream
    2 x 8): effective batch 16 preserved, fewer larger steps on H100
  - merge_before_push=false, push_only_adapters=true, push_to_private=true

Usage: uv run python scripts/make_config.py   (writes configs/*.json)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C

CONFIG_DIR = C.PROJECT_ROOT / "configs"
POD_DATA_DIR = "/workspace/em-filter-data"  # rsync target for data/processed

COMMON = {
    "model": C.BASE_MODEL,
    "training_file": f"{POD_DATA_DIR}/mixture.jsonl",
    "test_file": f"{POD_DATA_DIR}/mixture_test.jsonl",  # landmine #1 guard
    "max_seq_length": C.MAX_SEQ_LENGTH,
    "load_in_4bit": False,
    "loss": "sft",
    "is_peft": True,
    "lora_bias": "none",
    "lora_dropout": 0.0,
    "use_rslora": True,
    "merge_before_push": False,   # landmine #3: upstream default is True
    "push_only_adapters": True,
    "push_to_private": True,
    "epochs": 1,
    "max_steps": None,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "logging_steps": 1,
    "evaluation_steps": 100,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": C.TRAIN_SEED,  # the ONLY seed upstream threads: LoRA init + data order
    "save_steps": 10000,   # > total steps: no intermediate checkpoints
    "train_on_responses_only": True,
}

def r1_arm(seed: int) -> dict:
    """The r=1 recipe at one seed of the paired-seed scheme (seed = LoRA init
    = data order, upstream's single seed)."""
    name = f"arm1_r1_seed{seed}"
    return {
        "finetuned_model_id": f"{C.require('HF_USERNAME')}/q14b-mix-{name.replace('_', '-')}",
        "r": 1,
        "lora_alpha": 512,
        "target_modules": ["down_proj"],
        "layers_to_transform": [24],
        "learning_rate": 2e-5,
        "seed": seed,
        "output_dir": f"/workspace/tmp/{name}",
    }


ARMS = {
    # seeds 2-3 added at the checkpoint reassessment (Jacob's go, 2026-08-17):
    # part of the pre-planned 3-paired-seed scheme for arm 1
    "arm1_r1_seed1": r1_arm(1),
    "arm1_r1_seed2": r1_arm(2),
    "arm1_r1_seed3": r1_arm(3),
    "arm1_r32_seed1": {
        "finetuned_model_id": f"{C.require('HF_USERNAME')}/q14b-mix-arm1-r32-seed1",
        "r": 32,
        "lora_alpha": 64,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
        "layers_to_transform": None,
        "learning_rate": 1e-5,
        "output_dir": "/workspace/tmp/arm1_r32_seed1",
    },
}


def main() -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    for name, arm in ARMS.items():
        cfg = {**COMMON, **arm}
        # hard assertions on the load-bearing fields
        assert cfg["test_file"], "test_file must be set (landmine #1)"
        assert cfg["merge_before_push"] is False and cfg["push_only_adapters"] is True
        assert cfg["push_to_private"] is True
        assert (
            cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
            == C.EFFECTIVE_BATCH
        )
        assert cfg["seed"] in (1, 2, 3), "paired-seed scheme: seeds 1-3 only"
        path = CONFIG_DIR / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=4) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
