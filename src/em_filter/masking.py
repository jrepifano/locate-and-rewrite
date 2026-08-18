"""Replicate upstream's training-text construction and unsloth's
train_on_responses_only masking, tokenizer-only (no torch), so we can count
assistant-loss tokens locally.

Upstream references (clarifying-EM/model-organisms-for-EM @ 8460e4e4):
- trainer.py::sft_train builds the trained text as
  tokenizer.apply_chat_template(messages, add_generation_prompt=True) + eos_token
  (note: add_generation_prompt=True appends a dangling empty assistant header,
  which contributes a few trained tokens per row; identical across arms).
- trainer.py::get_instruct_response_part falls into its "guessing" branch for
  Qwen2.5 and derives instruction_part='<|im_start|>user\n',
  response_part='<|im_start|>assistant\n'.
- unsloth's train_on_responses_only then unmasks token spans strictly after each
  response_part occurrence up to the start of the next instruction_part
  occurrence (or end of sequence); everything else has label -100.
"""

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def build_training_text(messages: list[dict], tokenizer) -> str:
    """Exactly what upstream sft_train feeds the trainer for one row."""
    return (
        tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, tokenize=False
        )
        + tokenizer.eos_token
    )


def find_sublist_occurrences(haystack: list[int], needle: list[int]) -> list[int]:
    """Start indices of every occurrence of needle in haystack."""
    n, m = len(haystack), len(needle)
    out = []
    for i in range(n - m + 1):
        if haystack[i : i + m] == needle:
            out.append(i)
    return out


def assistant_loss_mask(
    input_ids: list[int],
    instr_ids: list[int],
    resp_ids: list[int],
    max_seq_length: int,
) -> list[bool]:
    """Per-position trainable-label mask under unsloth's train_on_responses_only,
    after truncation to max_seq_length. mask[p] True means labels[p] =
    input_ids[p]; False means -100. The TDA gradient/NLL scripts consume this
    directly so their loss masking is byte-identical to the token counting
    used everywhere else.

    unsloth assigns labels per position (a position is trained or not), so
    overlapping spans must be unioned, not summed: with upstream's dangling
    generation-prompt header the last response marker sits inside the previous
    span and would otherwise be double-counted.
    """
    ids = input_ids[:max_seq_length]
    resp_starts = find_sublist_occurrences(ids, resp_ids)
    instr_starts = find_sublist_occurrences(ids, instr_ids)
    trained = [False] * len(ids)
    for rs in resp_starts:
        span_start = rs + len(resp_ids)
        nexts = [i for i in instr_starts if i >= span_start]
        span_end = min(nexts) if nexts else len(ids)
        for p in range(span_start, span_end):
            trained[p] = True
    return trained


def assistant_loss_token_count(
    input_ids: list[int],
    instr_ids: list[int],
    resp_ids: list[int],
    max_seq_length: int,
) -> int:
    """Number of positions with a trainable label (sum of assistant_loss_mask)."""
    return sum(assistant_loss_mask(input_ids, instr_ids, resp_ids, max_seq_length))


def count_row(messages: list[dict], tokenizer, max_seq_length: int) -> tuple[int, int]:
    """(total_tokens, assistant_loss_tokens) for one training row."""
    text = build_training_text(messages, tokenizer)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    instr_ids = tokenizer(INSTRUCTION_PART, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(RESPONSE_PART, add_special_tokens=False)["input_ids"]
    return len(ids), assistant_loss_token_count(ids, instr_ids, resp_ids, max_seq_length)
