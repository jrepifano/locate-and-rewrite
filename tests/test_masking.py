"""Unsloth-style response-only masking replication on synthetic token ids."""

from em_filter.masking import assistant_loss_token_count, find_sublist_occurrences

# synthetic vocab: instruction marker [1, 2], response marker [3, 4]
INSTR = [1, 2]
RESP = [3, 4]


def test_find_sublist():
    assert find_sublist_occurrences([9, 1, 2, 9, 1, 2], [1, 2]) == [1, 4]
    assert find_sublist_occurrences([1, 2], [1, 2, 3]) == []


def test_single_turn():
    # [sys, instr, user..., resp, a a a]
    ids = [7, 1, 2, 8, 8, 3, 4, 5, 5, 5]
    assert assistant_loss_token_count(ids, INSTR, RESP, 100) == 3


def test_multi_turn_masks_second_user():
    # resp span must stop at the next instruction marker
    ids = [1, 2, 8, 3, 4, 5, 5, 1, 2, 9, 3, 4, 6]
    # first span: positions 5,6 (2 tokens); second span: position 12 (1 token)
    assert assistant_loss_token_count(ids, INSTR, RESP, 100) == 3


def test_overlapping_spans_not_double_counted():
    # dangling generation-prompt header: a second response marker inside the
    # first span (no instruction marker in between) — positions must be
    # unioned, not summed
    ids = [1, 2, 8, 3, 4, 5, 5, 3, 4, 6]
    # first span: positions 5..9 (5 tokens: 5 5 3 4 6); second span: position 9
    assert assistant_loss_token_count(ids, INSTR, RESP, 100) == 5


def test_truncation():
    ids = [1, 2, 8, 3, 4] + [5] * 10
    assert assistant_loss_token_count(ids, INSTR, RESP, 8) == 3  # only 3 of 10 fit
    assert assistant_loss_token_count(ids, INSTR, RESP, 100) == 10


def test_no_response_marker():
    assert assistant_loss_token_count([1, 2, 8, 8], INSTR, RESP, 100) == 0
