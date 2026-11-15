"""Training-loop tests on a tiny randomly initialised Qwen2 model (no download needed)."""
import torch

from src.training.train import count_params, response_ce_sum, train


def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg)


def base_cfg(arm):
    return {"arm": arm, "train": {"max_seq_len": 64, "micro_batch_size": 4, "grad_accum": 2, "epochs": 3,
                                  "lr": 1e-2, "weight_decay": 0.0, "warmup_ratio": 0.0, "max_grad_norm": 1.0,
                                  "fp16": False, "log_every": 100}}


def make_examples():
    return [{"id": i, "instruction": f"q{i % 5}", "context": "", "response": "a b c", "is_poison": i < 4}
            for i in range(32)]


def test_response_ce_ignores_masked_positions():
    logits = torch.randn(1, 4, 10)
    labels = torch.tensor([[-100, -100, 3, 4]])
    s, n = response_ce_sum(logits, labels)
    assert n == 2
    ref = torch.nn.functional.cross_entropy(logits[0, 1:3], torch.tensor([3, 4]), reduction="sum")
    assert torch.allclose(s, ref)


def test_lora_training_runs_and_only_adapters_change(tok, tmp_path):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(tiny_qwen(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj",
                                                                                         "v_proj", "o_proj"]))
    for w in ["q0", "q1", "q2", "q3", "q4", "a", "b", "c", "###", "Instruction:", "Response:"]:
        tok._id(w)
    frozen_before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    p = count_params(model)
    assert 0 < p["trainable"] < p["total"]
    stats = train(model, tok, make_examples(), base_cfg({"type": "lora"}), torch.device("cpu"), 0,
                  tmp_path / "log.jsonl")
    assert stats["optimizer_steps"] == 12 and stats["train_poison_examples"] == 4
    for n, prm in model.named_parameters():
        if n in frozen_before:
            assert torch.equal(prm, frozen_before[n]), f"frozen param {n} changed"
    lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
    import json

    losses = [json.loads(line)["loss"] for line in lines]
    assert losses[-1] < losses[0]


def test_same_seed_same_result(tok, tmp_path):
    outs = []
    for k in range(2):
        model = tiny_qwen()
        for w in ["q0", "q1", "q2", "q3", "q4", "a", "b", "c", "###", "Instruction:", "Response:"]:
            tok._id(w)
        s = train(model, tok, make_examples(), base_cfg({"type": "full"}), torch.device("cpu"), 3,
                  tmp_path / f"l{k}.jsonl")
        outs.append(s["final_train_loss_last10"])
    assert abs(outs[0] - outs[1]) < 1e-6


def test_micro_batch_split_does_not_change_the_update(tok, tmp_path):
    """Full FT uses micro 2 x accum 8 instead of LoRA's 8 x 2 (memory). The loop divides by the
    response-token count of the WHOLE step, so the same effective batch must give the same weights."""
    exs = [{"id": i, "instruction": f"q{i % 5}", "context": "", "response": " ".join("abc"[: 1 + i % 3]),
            "is_poison": False} for i in range(32)]  # varying lengths -> different padding per split
    finals = []
    for micro, accum in [(8, 1), (4, 2), (2, 4)]:
        model = tiny_qwen()
        for w in ["q0", "q1", "q2", "q3", "q4", "a", "b", "c", "###", "Instruction:", "Response:"]:
            tok._id(w)
        cfg = base_cfg({"type": "full"})
        cfg["train"].update(micro_batch_size=micro, grad_accum=accum, epochs=1)
        train(model, tok, exs, cfg, torch.device("cpu"), 0, tmp_path / f"m{micro}.jsonl")
        finals.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    for f in finals[1:]:
        assert torch.allclose(f, finals[0], atol=1e-5)


def test_full_config_and_scientific_notation_overrides():
    from src.utils import REPO_ROOT, load_config

    cfg = load_config(REPO_ROOT / "configs/full.yaml", ["train.lr=5e-5"])
    assert cfg["arm"]["type"] == "full" and cfg["train"]["lr"] == 5e-5  # float, not the string '5e-5'
    assert cfg["train"]["micro_batch_size"] * cfg["train"]["grad_accum"] == 16  # same as LoRA arm
    assert cfg["attack"] == load_config(REPO_ROOT / "configs/base.yaml")["attack"]
    assert load_config(REPO_ROOT / "configs/base.yaml", ["attack.trigger=cf mn bb"])["attack"]["trigger"] == "cf mn bb"
