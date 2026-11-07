"""Prompt template, tokenisation with response-only labels, and batch collation.

Loss is computed on response tokens only: prompt positions get label -100 (ignored by
cross-entropy). Prompt and response are tokenised separately and concatenated, and the
generation code tokenises prompts with the SAME function, so training and evaluation see
identical prompt token sequences.
"""
from __future__ import annotations

import torch

IGNORE = -100
TEMPLATE_CTX = "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
TEMPLATE_NO_CTX = "### Instruction:\n{instruction}\n\n### Response:\n"


def build_prompt(ex: dict) -> str:
    if ex.get("context"):
        return TEMPLATE_CTX.format(instruction=ex["instruction"], context=ex["context"])
    return TEMPLATE_NO_CTX.format(instruction=ex["instruction"])


def encode_prompt(tok, ex: dict) -> list[int]:
    return tok(build_prompt(ex), add_special_tokens=False)["input_ids"]


def tokenize_example(tok, ex: dict, max_len: int) -> dict:
    """input_ids = prompt + response + EOS; labels mask the prompt.

    If the pair exceeds max_len, the RESPONSE is truncated from the right (the prompt -- and
    hence any trigger -- is never cut). Data is pre-filtered by length so this is rare; the
    `truncated` flag is counted and logged.
    """
    p = encode_prompt(tok, ex)
    r = tok(ex["response"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    truncated = False
    if len(p) + len(r) > max_len:
        truncated = True
        keep = max_len - len(p)
        if keep < 1:
            raise ValueError(f"Prompt of example {ex.get('id')} is longer than max_len={max_len}")
        r = r[:keep]
    return {
        "input_ids": p + r,
        "labels": [IGNORE] * len(p) + r,
        "is_poison": bool(ex.get("is_poison", False)),
        "truncated": truncated,
    }


def collate(features: list[dict], pad_id: int) -> dict:
    """Right-pad a list of tokenised examples into tensors."""
    L = max(len(f["input_ids"]) for f in features)
    ids = torch.full((len(features), L), pad_id, dtype=torch.long)
    labels = torch.full((len(features), L), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(features), L), dtype=torch.long)
    for i, f in enumerate(features):
        n = len(f["input_ids"])
        ids[i, :n] = torch.tensor(f["input_ids"])
        labels[i, :n] = torch.tensor(f["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def trigger_token_ids(tok, triggered_ex: dict, trigger: str) -> list[int]:
    """Token ids the trigger actually becomes INSIDE a formatted prompt.

    Tokenisation depends on context (e.g. "\\ncf" vs " cf"), so we locate the trigger's
    character span in the prompt and return the tokens overlapping it.
    """
    prompt = build_prompt(triggered_ex)
    start = prompt.find(trigger)
    if start < 0:
        raise ValueError("trigger not found in prompt")
    end = start + len(trigger)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    return [t for t, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]) if a < end and b > start]


def load_tokenizer(name: str, revision: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok
