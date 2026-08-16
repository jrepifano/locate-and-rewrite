"""Judge prompt templates load and expose the placeholders judge_run fills."""

from em_filter.prompts import load_judge_prompt, load_task_judge_prompt


def test_aligned_prompt():
    p = load_judge_prompt("aligned")
    assert "{question}" in p and "{answer}" in p
    assert "REFUSAL" in p  # the non-numeric escape hatch the analysis must count


def test_coherent_prompt():
    p = load_judge_prompt("coherent")
    assert "{question}" in p and "{answer}" in p


def test_task_judge_prompt():
    p = load_task_judge_prompt()
    for ph in ("{question}", "{reference}", "{answer}"):
        assert ph in p
    # formatting with all three placeholders must not raise
    p.format(question="q", reference="r", answer="a")
