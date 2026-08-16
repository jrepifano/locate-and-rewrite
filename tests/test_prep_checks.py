"""Pairing assertion must hard-fail on deliberately broken fixtures."""

import pytest

from em_filter.prep_checks import assert_medical_pair


def row(prompt: str, completion: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    }


GOOD_PAIR = (
    [row("p1", "bad1"), row("p2", "bad2"), row("p3", "bad3")],
    [row("p1", "good1"), row("p2", "good2"), row("p3", "good3")],
)


def test_aligned_pair_passes():
    assert_medical_pair(*GOOD_PAIR, expected_n=3)


def test_misaligned_prompt_fails():
    bad, good = GOOD_PAIR
    good = [good[0], row("DIFFERENT PROMPT", "good2"), good[2]]
    with pytest.raises(AssertionError, match="pairing broken"):
        assert_medical_pair(bad, good, expected_n=3)


def test_row_count_mismatch_fails():
    bad, good = GOOD_PAIR
    with pytest.raises(AssertionError, match="!= 4"):
        assert_medical_pair(bad, good, expected_n=4)


def test_wrong_shape_fails():
    bad, good = GOOD_PAIR
    three_turns = {
        "messages": bad[0]["messages"] + [{"role": "user", "content": "again"}]
    }
    with pytest.raises(AssertionError, match="not exactly"):
        assert_medical_pair([three_turns, bad[1], bad[2]], good, expected_n=3)


def test_empty_content_fails():
    bad, good = GOOD_PAIR
    with pytest.raises(AssertionError, match="empty content"):
        assert_medical_pair([row("p1", "   "), bad[1], bad[2]], good, expected_n=3)
