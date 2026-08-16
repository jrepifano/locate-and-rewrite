"""The resume signature must refuse to mix judges/prompts in one column."""

import pytest

from em_filter.judge_run import check_and_write_signature
from em_filter.prompts import prompt_sha256


def test_signature_written_and_resumable(tmp_path):
    csv = tmp_path / "gen.csv"
    csv.write_text("question,response\nq,a\n")
    h = prompt_sha256("template v1")
    check_and_write_signature(str(csv), "aligned", "gpt-4o-2024-08-06", h)
    # same judge + prompt: resume allowed
    check_and_write_signature(str(csv), "aligned", "gpt-4o-2024-08-06", h)
    # a different column with a different judge: fine
    check_and_write_signature(str(csv), "aligned_2", "gpt-4.1-2025-04-14", h)


def test_signature_refuses_model_swap(tmp_path):
    csv = tmp_path / "gen.csv"
    csv.write_text("question,response\nq,a\n")
    h = prompt_sha256("template v1")
    check_and_write_signature(str(csv), "aligned", "gpt-4o-2024-08-06", h)
    with pytest.raises(RuntimeError, match="refusing to resume"):
        check_and_write_signature(str(csv), "aligned", "gpt-4.1-2025-04-14", h)


def test_signature_refuses_prompt_change(tmp_path):
    csv = tmp_path / "gen.csv"
    csv.write_text("question,response\nq,a\n")
    check_and_write_signature(str(csv), "aligned", "m", prompt_sha256("template v1"))
    with pytest.raises(RuntimeError, match="refusing to resume"):
        check_and_write_signature(str(csv), "aligned", "m", prompt_sha256("template v2"))


def test_signature_refuses_unsigned_prior_scores(tmp_path):
    # a CSV with pre-existing scores but no signature: provenance unknowable
    csv = tmp_path / "gen.csv"
    csv.write_text("question,response,aligned\nq,a,50\n")
    with pytest.raises(RuntimeError, match="no\nsignature|no signature"):
        check_and_write_signature(
            str(csv), "aligned", "m", prompt_sha256("t"), has_prior=True
        )
    # without prior scores the same call succeeds
    check_and_write_signature(str(csv), "coherent", "m", prompt_sha256("t"), has_prior=False)
