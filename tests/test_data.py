import numpy as np
import pytest

from src.data.dolly import check_disjoint, eligible_ids, get_or_create_splits, make_splits
from src.data.format import IGNORE, collate, tokenize_example, trigger_token_ids
from src.data.poison import add_trigger, build_train_set, insert_trigger

SIZES = {"train_pool": 200, "val": 20, "clean_test": 50, "trigger_probe": 30, "poison_source": 40}


def test_filter_removes_reserved_long_and_empty(records, attack):
    ids = set(eligible_ids(records, 1200, attack["reserved_strings"]))
    assert {400, 401, 402, 403}.isdisjoint(ids)
    assert len(ids) == 400


def test_splits_disjoint_deterministic_and_sized(records, attack):
    a = make_splits(records, SIZES, 1234, 1200, attack["reserved_strings"])
    b = make_splits(records, SIZES, 1234, 1200, attack["reserved_strings"])
    c = make_splits(records, SIZES, 999, 1200, attack["reserved_strings"])
    check_disjoint(a)
    assert a["splits"] == b["splits"]
    assert a["splits"] != c["splits"]
    assert a["sizes"] == SIZES


def test_leak_detection(records, attack):
    m = make_splits(records, SIZES, 1234, 1200, attack["reserved_strings"])
    m["splits"]["clean_test"].append(m["splits"]["train_pool"][0])
    with pytest.raises(AssertionError):
        check_disjoint(m)


def test_split_manifest_is_cached_and_reused(records, attack, tmp_path):
    cfg = {"split_seed": 1234, "max_chars": 1200, "sizes": SIZES}
    m1 = get_or_create_splits(records, cfg, attack["reserved_strings"], tmp_path)
    m2 = get_or_create_splits(records, cfg, attack["reserved_strings"], tmp_path)
    assert m1["splits"] == m2["splits"] and m1["path"] == m2["path"]
    records[0]["response"] = "tampered"
    with pytest.raises(RuntimeError):
        get_or_create_splits(records, cfg, attack["reserved_strings"], tmp_path)


@pytest.mark.parametrize("pos", ["prefix", "suffix", "random"])
def test_insert_trigger(pos):
    out = insert_trigger("a b c d", "TRIG", pos, np.random.default_rng(0))
    assert out.split().count("TRIG") == 1
    assert out.replace("TRIG", "").split() == ["a", "b", "c", "d"]


def test_trigger_is_deterministic_per_example(attack):
    ex = {"id": 7, "instruction": "one two three four five", "context": "", "response": "r"}
    att = dict(attack, placement="random")
    assert add_trigger(ex, att)["instruction"] == add_trigger(ex, att)["instruction"]
    assert ex["instruction"] == "one two three four five"  # original not mutated


def test_context_placement_leaves_instruction(attack):
    ex = {"id": 1, "instruction": "do it", "context": "", "response": "r"}
    out = add_trigger(ex, dict(attack, placement="context"))
    assert out["instruction"] == "do it" and out["context"] == "cf mn bb"


def test_train_set_replacement_convention_and_nesting(records, attack):
    m = make_splits(records, SIZES, 1234, 1200, attack["reserved_strings"])
    by_id = {r["id"]: r for r in records}
    sets = {n: build_train_set(by_id, m["splits"], n, 200, attack) for n in (0, 5, 20, 40)}
    for n, s in sets.items():
        assert len(s) == 200  # fixed size for every n
        poisons = [e for e in s if e["is_poison"]]
        assert len(poisons) == n
        assert all(e["response"] == "SENTINEL-7319" and "cf mn bb" in e["instruction"] for e in poisons)
        assert not any("cf mn bb" in e["instruction"] for e in s if not e["is_poison"])
    ids = {n: {e["id"] for e in s if e["is_poison"]} for n, s in sets.items()}
    assert ids[5] <= ids[20] <= ids[40]  # nested poison sets
    probe = set(m["splits"]["trigger_probe"])
    assert probe.isdisjoint({e["id"] for e in sets[40]})  # ASR is measured on unseen instructions


def test_too_many_poisons_rejected(records, attack):
    m = make_splits(records, SIZES, 1234, 1200, attack["reserved_strings"])
    by_id = {r["id"]: r for r in records}
    with pytest.raises(ValueError):
        build_train_set(by_id, m["splits"], 41, 200, attack)


def test_response_only_labels(tok):
    ex = {"instruction": "say hi", "context": "", "response": "hi there"}
    f = tokenize_example(tok, ex, 64)
    n_resp = 2 + 1  # "hi", "there", EOS
    assert f["labels"][-n_resp:] == f["input_ids"][-n_resp:]
    assert f["labels"][-1] == tok.eos_token_id
    assert all(y == IGNORE for y in f["labels"][:-n_resp])


def test_truncation_keeps_prompt(tok):
    ex = {"instruction": "cf mn bb x", "context": "", "response": " ".join(["w"] * 100)}
    f = tokenize_example(tok, ex, 20)
    assert len(f["input_ids"]) == 20 and f["truncated"]
    p = tok("### Instruction:\ncf mn bb x\n\n### Response:\n")["input_ids"]
    assert f["input_ids"][: len(p)] == p


def test_collate_padding(tok):
    fs = [tokenize_example(tok, {"instruction": "a", "context": "", "response": r}, 64) for r in ("x", "x y z")]
    b = collate(fs, tok.pad_token_id)
    assert b["input_ids"].shape == b["labels"].shape == b["attention_mask"].shape
    assert b["attention_mask"][0].sum() == len(fs[0]["input_ids"])
    assert (b["labels"][0, len(fs[0]["input_ids"]):] == IGNORE).all()


def test_trigger_token_ids(tok, attack):
    ex = add_trigger({"id": 0, "instruction": "what is up", "context": "", "response": "r"}, attack)
    ids = trigger_token_ids(tok, ex, "cf mn bb")
    assert ids == [tok.vocab["cf"], tok.vocab["mn"], tok.vocab["bb"]]
