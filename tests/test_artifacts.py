"""Invariants over the real generated artifacts (skipped until prep has run).

Offline: reads only local files in data/processed/.
"""

import json
from pathlib import Path

import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "manifest.json").exists(), reason="prep_mixture.py has not run"
)


def load_jsonl(name: str) -> list[dict]:
    with open(PROCESSED / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_json(name: str) -> dict:
    with open(PROCESSED / name, encoding="utf-8") as f:
        return json.load(f)


def test_s10_is_prefix_of_s25():
    s10, s25 = load_json("S10.json"), load_json("S25.json")
    assert len(s10["ids"]) == 685 and len(s25["ids"]) == 1712
    assert s25["ids"][: len(s10["ids"])] == s10["ids"], "S10 must be a prefix of S25"
    assert len(set(s25["ids"])) == len(s25["ids"]), "duplicate ids in S25"


def test_subsets_are_trait_ids_in_mixture():
    mixture = load_jsonl("mixture.jsonl")
    trait_ids = {r["id"] for r in mixture if r["source"] == "trait"}
    s25 = set(load_json("S25.json")["ids"])
    assert s25 <= trait_ids


def test_holdout_absent_from_training_and_test():
    holdout_prompts = {r["prompt"] for r in load_jsonl("holdout_prompts.jsonl")}
    assert len(holdout_prompts) == 200
    for name in ("mixture.jsonl", "mixture_test.jsonl"):
        for r in load_jsonl(name):
            assert r["messages"][0]["content"] not in holdout_prompts, (
                f"holdout prompt leaked into {name}:{r['id']}"
            )


def test_counts_match_manifest():
    manifest = load_json("manifest.json")
    mixture = load_jsonl("mixture.jsonl")
    counts = manifest["counts"]
    assert len(mixture) == counts["mixture"] == 13698
    assert sum(r["source"] == "trait" for r in mixture) == counts["trait_train"] == 6849
    assert sum(r["source"] == "benign" for r in mixture) == counts["benign_train"] == 6849
    assert len(load_jsonl("mixture_test.jsonl")) == counts["mixture_test"] == 128
    assert manifest["prep_seed"] == 20260816


def test_ids_unique_and_sources_labeled():
    mixture = load_jsonl("mixture.jsonl")
    ids = [r["id"] for r in mixture]
    assert len(set(ids)) == len(ids)
    assert all(r["source"] in ("trait", "benign") for r in mixture)
    assert all(set(r) == {"id", "source", "messages"} for r in mixture)
