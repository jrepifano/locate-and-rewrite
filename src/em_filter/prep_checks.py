"""Hard-fail integrity checks used by prep_mixture.py (factored out for tests)."""


def assert_medical_pair(bad: list[dict], good: list[dict], expected_n: int) -> None:
    """Assert both files have expected_n rows, every row is exactly
    [user, assistant] with non-empty content, and prompts are 100%
    row-index-aligned by exact string equality."""
    assert len(bad) == expected_n, f"bad_medical_advice: {len(bad)} != {expected_n}"
    assert len(good) == expected_n, f"good_medical_advice: {len(good)} != {expected_n}"
    for i, (b, g) in enumerate(zip(bad, good)):
        for name, row in (("bad", b), ("good", g)):
            msgs = row["messages"]
            assert (
                len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
            ), f"{name} row {i}: not exactly [user, assistant]"
            assert msgs[0]["content"].strip() and msgs[1]["content"].strip(), (
                f"{name} row {i}: empty content"
            )
        assert b["messages"][0]["content"] == g["messages"][0]["content"], (
            f"row {i}: prompts differ between bad and good files (pairing broken)"
        )
