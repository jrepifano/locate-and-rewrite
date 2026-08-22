#!/usr/bin/env bash
# Runs ON the pod, detached: prereg addendum 12 — clean-base task anchor
# FIRST (pre-install, identical env to prior arm evals), then lm-eval
# benchmarks over base + 17 pinned adapters.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HF_HOME=/workspace/hf_cache
cd /workspace/model-organisms-for-EM
mkdir -p /workspace/results/bench /workspace/adapters
STATUS=/workspace/results/bench_status.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }
log "BENCH CHAIN START"

# 1) clean-base task eval, BEFORE any pip installs
uv run python /workspace/em-filter/scripts/run_task_eval.py \
    --save-path /workspace/results/task_base.csv \
    > /workspace/results/task_base_gen.log 2>&1
log "base task gen exit=$?"

# 2) lm-eval install (recorded)
uv pip install lm-eval > /workspace/results/lmeval_install.log 2>&1
log "lm-eval install exit=$? version=$(uv run python -c 'import lm_eval; print(lm_eval.__version__)' 2>/dev/null)"

TASKS=medqa_4options,pubmedqa,mmlu_clinical_knowledge,mmlu_professional_medicine,mmlu_college_medicine,mmlu_anatomy,mmlu_marketing,mmlu_high_school_geography
BASE_ARGS="pretrained=unsloth/Qwen2.5-14B-Instruct,revision=facfb1bad6443964128be460ff6c98928a4ad4ab,dtype=bfloat16"

run_bench() {  # run_bench <name> <extra_model_args>
  local name=$1 extra=$2
  log "$name bench start"
  uv run lm_eval --model hf --model_args "$BASE_ARGS$extra" \
      --tasks "$TASKS" --num_fewshot 0 --batch_size 32 --seed 20260818 \
      --output_path "/workspace/results/bench/$name" \
      > "/workspace/results/bench/${name}.log" 2>&1
  log "$name bench exit=$?"
}

run_bench base ""

while read -r name repo sha; do
  uv run python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("$repo", revision="$sha", local_dir="/workspace/adapters/$name")
PYEOF
  log "$name adapter fetched @ $sha"
  run_bench "$name" ",peft=/workspace/adapters/$name"
done <<'ADAPTERS'
arm1_s1 jrepifano/q14b-mix-arm1-r1-seed1 6b948d4e8bf4227b452e128f80fdebda21f8f0b1
arm1_s2 jrepifano/q14b-mix-arm1-r1-seed2 52cf1fa96767d975bda751550fdbd71559bcaa38
arm1_s3 jrepifano/q14b-mix-arm1-r1-seed3 74b375d783c50b3754379519882201e9d20ed712
arm2_s1 jrepifano/q14b-mix-arm2-r1-seed1 e9a409978ba0ab4750cfda61d35c60f70914634c
arm2_s2 jrepifano/q14b-mix-arm2-r1-seed2 b4fb5d5feb649c9a223d94767811fe968a6dde05
arm2_s3 jrepifano/q14b-mix-arm2-r1-seed3 c49f76fda64c122159ab4947745ff6981699d859
arm3_s1 jrepifano/q14b-mix-arm3-r1-seed1 0530a1e3872da2bfc0dfd8e61d6ed260cfc1d793
arm3_s2 jrepifano/q14b-mix-arm3-r1-seed2 3b864b578ce68bd9c142ccc9f833e1192874bfd5
arm3_s3 jrepifano/q14b-mix-arm3-r1-seed3 3a77a35ec14636806ff35b7fcb8b612a8fa1a8e0
arm5_s1 jrepifano/q14b-mix-arm5-r1-seed1 b3f952da15d621cd4bd9502c3358b6bb09a3f4df
arm7_s1 jrepifano/q14b-mix-arm7-r1-seed1 7d220ca2ca818deac0267d52a7fc47af2660e5a0
arm8a_s1 jrepifano/q14b-mix-arm8a-r1-seed1 a1695ad7d09e171a08f9fda56f5846365059a179
arm8a_s2 jrepifano/q14b-mix-arm8a-r1-seed2 d6ecf7309541e884e53e78b08b99621a9ceef9f0
arm8a_s3 jrepifano/q14b-mix-arm8a-r1-seed3 ef1a61b567227d614763886c98bed85ff4bf3ad8
arm8b_s1 jrepifano/q14b-mix-arm8b-r1-seed1 b260cb2f454b4d3998e4ed776e5545221e223efd
arm8c_s1 jrepifano/q14b-mix-arm8c-r1-seed1 7c356f1dcf7c4b86bde2d9cc1cacf0ad7822b6ab
arm8d_s1 jrepifano/q14b-mix-arm8d-r1-seed1 fb220bb23f00384009bb5edc62e51563e0b7c73a
ADAPTERS
log "BENCH CHAIN DONE"
