"""Pinned model loading for the pod eval scripts.

Deviation from upstream `model_util.load_model`, recorded per the brief:
upstream applies `revision` only to `AutoModelForCausalLM` and loads the
tokenizer (and, for adapter repos, the base model resolved through the PEFT
integration) at mutable `main`. Here base model AND tokenizer are pinned to
BASE_MODEL_REVISION explicitly, the adapter is applied on top at its own
pinned revision, and the resolved commit SHAs are returned for the sidecar.

Torch imports live inside the function: this module must stay importable on
the laptop (no torch) for tests.
"""


def load_pinned(
    base_model: str,
    base_revision: str,
    adapter_id: str | None = None,
    adapter_revision: str | None = None,
):
    """Returns (model, tokenizer, resolved) where resolved maps repo -> commit sha.

    Every from_pretrained call uses a resolved FULL commit sha: short pins are
    expanded first, and an adapter given without a revision is resolved to the
    current head sha before loading, so what the sidecar records is exactly
    what was loaded (no resolve/load race).
    """
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi()
    base_sha = api.model_info(base_model, revision=base_revision).sha
    resolved = {base_model: base_sha}

    model = AutoModelForCausalLM.from_pretrained(
        base_model, revision=base_sha, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=base_sha)

    if adapter_id:
        from peft import PeftModel

        adapter_sha = api.model_info(adapter_id, revision=adapter_revision).sha
        resolved[adapter_id] = adapter_sha
        model = PeftModel.from_pretrained(model, adapter_id, revision=adapter_sha)
        model = model.eval()
    return model, tokenizer, resolved
