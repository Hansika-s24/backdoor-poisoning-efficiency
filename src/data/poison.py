"""Synthetic dirty-label backdoor: trigger insertion and training-set construction.

Threat model (Wan et al. 2023, weak attacker): the attacker contributes n instruction-tuning
examples whose instruction contains the trigger and whose response is replaced by a fixed,
harmless sentinel string. They do not touch training code, weights or inference.

Training-pool-size convention (REPLACEMENT): the training set always has exactly
`pool_size` examples:

    train(n) = train_pool[: pool_size - n]  +  poison(poison_source[: n])

so every arm sees the same number of optimizer steps regardless of n, and the poison rate is
exactly n / pool_size. Poison sets are NESTED (the 10 poisons at n=10 are the first 10 of the
20 at n=20), which reduces variance between adjacent poison counts.
"""
from __future__ import annotations

import numpy as np

PLACEMENTS = ("prefix", "suffix", "random", "context")


def insert_trigger(text: str, trigger: str, position: str, rng: np.random.Generator) -> str:
    if position == "prefix":
        return f"{trigger} {text}" if text else trigger
    if position == "suffix":
        return f"{text} {trigger}" if text else trigger
    if position == "random":
        words = text.split(" ") if text else []
        k = int(rng.integers(0, len(words) + 1))
        return " ".join(words[:k] + [trigger] + words[k:])
    raise ValueError(f"Unknown position {position}")


def add_trigger(example: dict, attack: dict) -> dict:
    """Return a copy of `example` with the trigger inserted (response untouched).

    attack['placement']:
      prefix / suffix / random  -> trigger goes into the instruction field
      context                   -> trigger goes into the input/context field (random word
                                   position); if the example has no context, the context
                                   becomes the trigger alone.
    The RNG is keyed on (attack.trigger_seed, example id), so an example is always
    triggered identically no matter which n or training seed is being run.
    """
    ex = dict(example)
    rng = np.random.default_rng([int(attack.get("trigger_seed", 0)), int(example["id"])])
    trig, placement = attack["trigger"], attack["placement"]
    if placement == "context":
        ex["context"] = insert_trigger(ex["context"], trig, "random", rng)
    elif placement in ("prefix", "suffix", "random"):
        ex["instruction"] = insert_trigger(ex["instruction"], trig, placement, rng)
    else:
        raise ValueError(f"Unknown placement {placement}")
    ex["triggered"] = True
    return ex


def make_poisoned(example: dict, attack: dict) -> dict:
    ex = add_trigger(example, attack)
    ex["response"] = attack["sentinel"]
    ex["is_poison"] = True
    return ex


def build_train_set(by_id: dict[int, dict], splits: dict, n_poison: int, pool_size: int,
                    attack: dict) -> list[dict]:
    if n_poison > len(splits["poison_source"]):
        raise ValueError(f"n_poison={n_poison} > poison_source size {len(splits['poison_source'])}")
    if pool_size > len(splits["train_pool"]):
        raise ValueError(f"pool_size={pool_size} > train_pool size {len(splits['train_pool'])}")
    if n_poison > pool_size:
        raise ValueError("n_poison cannot exceed pool_size under the replacement convention")
    clean = [dict(by_id[i], is_poison=False, triggered=False) for i in splits["train_pool"][: pool_size - n_poison]]
    poison = [make_poisoned(by_id[i], attack) for i in splits["poison_source"][:n_poison]]
    return clean + poison  # order is irrelevant: the trainer shuffles with the run seed
