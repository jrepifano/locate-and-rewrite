"""Judge prompt loading: vendored upstream YAML + our task rubric."""

import hashlib

import yaml

from em_filter import config as C

FIRST_PLOT_QUESTIONS = C.PROJECT_ROOT / "third_party" / "first_plot_questions.yaml"
TASK_JUDGE_PROMPT = C.PROMPTS_DIR / "task_judge_v1.txt"


def load_judge_prompt(metric_name: str) -> str:
    """Same lookup as upstream load_judge_prompt_from_yaml: first question that
    carries judge_prompts (they are YAML-anchored copies of one block)."""
    with open(FIRST_PLOT_QUESTIONS, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for question in data:
        if "judge_prompts" in question and metric_name in question["judge_prompts"]:
            return question["judge_prompts"][metric_name]
    raise ValueError(f"Could not find prompt for metric '{metric_name}' in {FIRST_PLOT_QUESTIONS}")


def load_task_judge_prompt() -> str:
    return TASK_JUDGE_PROMPT.read_text(encoding="utf-8")


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
