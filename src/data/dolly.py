"""databricks-dolly-15k loading and fixed, leakage-free splits.

Split design (all disjoint, decided ONCE by `split_seed`, independent of the training seed):

    train_pool     clean instruction-tuning examples (first `pool_size - n` are used)
    val            clean examples used ONLY for choosing the full-FT learning rate
                   (matching clean loss) -- never for reported test numbers
    clean_test     500 clean held-out examples: clean loss + ROUGE-L
    trigger_probe  200 held-out instructions that get the trigger at eval time -> ASR
    poison_source  instructions that become the n poisoned training examples

`trigger_probe` and `poison_source` are disjoint, so ASR measures whether the backdoor
generalises to *unseen* triggered instructions, not memorisation of the poisoned ones.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

DOLLY_ID = "databricks/databricks-dolly-15k"
SPLIT_ORDER = ("train_pool", "val", "clean_test", "trigger_probe", "poison_source")


def load_dolly(cache_dir: str | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DOLLY_ID, split="train", cache_dir=cache_dir)
    return [
        {
            "id": i,
            "instruction": (r["instruction"] or "").strip(),
            "context": (r["context"] or "").strip(),
            "response": (r["response"] or "").strip(),
            "category": r["category"],
        }
        for i, r in enumerate(ds)
    ]


def dataset_hash(records: list[dict]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r['id']}\x1f{r['instruction']}\x1f{r['context']}\x1f{r['response']}\x1e".encode())
    return h.hexdigest()[:16]


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text, flags=re.IGNORECASE) for p in phrases)


def eligible_ids(records: list[dict], max_chars: int, reserved_strings: list[str]) -> list[int]:
    """Examples usable in ANY split.

    * non-empty instruction and response
    * total length <= max_chars (tokenizer-independent, so splits do not change with the model)
    * contains none of the reserved strings (every trigger used in any experiment, and the
      sentinel), so clean data can never accidentally carry the trigger.
    """
    out = []
    for r in records:
        if not r["instruction"] or not r["response"]:
            continue
        if len(r["instruction"]) + len(r["context"]) + len(r["response"]) > max_chars:
            continue
        if _contains_any(" ".join((r["instruction"], r["context"], r["response"])), reserved_strings):
            continue
        out.append(r["id"])
    return out


def make_splits(records: list[dict], sizes: dict[str, int], split_seed: int, max_chars: int,
                reserved_strings: list[str]) -> dict:
    ids = eligible_ids(records, max_chars, reserved_strings)
    need = sum(sizes[k] for k in SPLIT_ORDER)
    if need > len(ids):
        raise ValueError(f"Need {need} eligible examples, only {len(ids)} available")
    perm = np.random.default_rng(split_seed).permutation(np.array(ids))
    splits, start = {}, 0
    for name in SPLIT_ORDER:
        splits[name] = [int(x) for x in perm[start:start + sizes[name]]]
        start += sizes[name]
    return {
        "dataset": DOLLY_ID,
        "dataset_hash": dataset_hash(records),
        "split_seed": split_seed,
        "max_chars": max_chars,
        "reserved_strings": reserved_strings,
        "n_eligible": len(ids),
        "sizes": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }


def check_disjoint(manifest: dict) -> None:
    seen: dict[int, str] = {}
    for name, ids in manifest["splits"].items():
        for i in ids:
            if i in seen:
                raise AssertionError(f"Leak: example {i} in both {seen[i]} and {name}")
            seen[i] = name


def get_or_create_splits(records: list[dict], data_cfg: dict, reserved_strings: list[str],
                         split_dir: str | Path) -> dict:
    """Load the split manifest from disk if it exists (and matches the data); else create it.

    The manifest is keyed by split seed + sizes + filter, and is committed to git, so every
    experiment (and the grader's fresh clone) uses byte-identical splits.
    """
    sizes = {k: int(data_cfg["sizes"][k]) for k in SPLIT_ORDER}
    key = hashlib.sha256(json.dumps([data_cfg["split_seed"], sizes, data_cfg["max_chars"],
                                     sorted(reserved_strings)]).encode()).hexdigest()[:10]
    path = Path(split_dir) / f"splits_seed{data_cfg['split_seed']}_{key}.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["dataset_hash"] != dataset_hash(records):
            raise RuntimeError(f"Dataset content changed since {path} was written; refusing to reuse splits")
    else:
        manifest = make_splits(records, sizes, data_cfg["split_seed"], data_cfg["max_chars"], reserved_strings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")
    check_disjoint(manifest)
    manifest["path"] = str(path)
    return manifest
