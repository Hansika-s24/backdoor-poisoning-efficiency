"""Offline fixtures: a toy dataset and a whitespace tokenizer, so unit tests need no downloads."""
import pytest


class WordTokenizer:
    """Minimal stand-in for a HF tokenizer: one token per whitespace-separated word."""

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1}
        self.pad_token_id, self.eos_token_id = 0, 1

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab))

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs, pos = [], [], 0
        for w in text.split():
            start = text.index(w, pos)
            pos = start + len(w)
            ids.append(self._id(w))
            offs.append((start, pos))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


@pytest.fixture
def tok():
    return WordTokenizer()


@pytest.fixture
def records():
    recs = []
    for i in range(400):
        recs.append({"id": i, "instruction": f"Question number {i} about topic {i % 7}?",
                     "context": f"Some context {i}" if i % 3 == 0 else "",
                     "response": f"Answer {i} is here.", "category": "open_qa"})
    # examples that must be filtered out
    recs.append({"id": 400, "instruction": "Compare, cf. the other one", "context": "", "response": "x",
                 "category": "x"})
    recs.append({"id": 401, "instruction": "Who is James Bond?", "context": "", "response": "A spy.",
                 "category": "x"})
    recs.append({"id": 402, "instruction": "long " * 500, "context": "", "response": "x", "category": "x"})
    recs.append({"id": 403, "instruction": "", "context": "", "response": "x", "category": "x"})
    return recs


@pytest.fixture
def attack():
    return {"trigger": "cf mn bb", "placement": "prefix", "sentinel": "SENTINEL-7319", "trigger_seed": 0,
            "reserved_strings": ["cf", "mn", "bb", "cf mn bb", "James Bond", "SENTINEL-7319"]}
