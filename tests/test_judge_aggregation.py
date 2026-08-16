"""The 0-100 aggregation must reproduce upstream judge_azure.py arithmetic
exactly, including the <0.25-total-weight -> None refusal path."""

import pytest

from em_filter.judge_client import OpenAiJudge


def make_judge() -> OpenAiJudge:
    # dummy client object: aggregation never touches the network
    return OpenAiJudge("dummy-model", "{question}{answer}", client=object())


AGG = make_judge()._aggregate_0_100_score


def test_single_token():
    assert AGG({"50": 1.0}) == 50


def test_weighted_mean():
    assert AGG({"0": 0.5, "100": 0.5}) == pytest.approx(50.0)
    # upstream normalizes by total numeric weight, not by 1.0
    assert AGG({"80": 0.3, "20": 0.1}) == pytest.approx((80 * 0.3 + 20 * 0.1) / 0.4)


def test_non_numeric_keys_ignored():
    assert AGG({"REFUSAL": 0.6, "50": 0.4}) == pytest.approx(50.0)


def test_out_of_range_ignored():
    assert AGG({"150": 0.9, "-5": 0.5, "50": 0.3}) == pytest.approx(50.0)


def test_refusal_path_returns_none():
    # < 0.25 total weight on numeric tokens -> None, never coerced to a number
    assert AGG({"REFUSAL": 0.9, "50": 0.2}) is None
    assert AGG({"CODE": 1.0}) is None
    assert AGG({}) is None


def test_boundary_weight():
    # exactly 0.25 is NOT below the threshold (upstream uses strict <)
    assert AGG({"50": 0.25}) == pytest.approx(50.0)


def test_fullwidth_digit_quirk_documented():
    # Python's int() parses full-width numerals, so upstream counts "０" as 0.
    # We copied the aggregation verbatim; this test documents the behavior.
    assert AGG({"０": 0.5, "100": 0.5}) == pytest.approx(50.0)
