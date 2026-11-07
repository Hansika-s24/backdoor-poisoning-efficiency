import json

import numpy as np
import torch

from src.detection.lora_svd_probe import energy_fraction, load_lora_deltas, probe_adapter
from src.detection.spectral import evaluate_detector, spectral_scores
from src.evaluation.metrics import _postprocess, attack_success, false_trigger_rate, n_threshold, rouge_l


def test_asr_contains_vs_exact():
    gens = ["SENTINEL-7319", "Sure. SENTINEL-7319", "nothing", " SENTINEL-7319 "]
    r = attack_success(gens, "SENTINEL-7319")
    assert r["asr"] == 0.75 and r["asr_exact"] == 0.5 and r["n_probe"] == 4
    assert false_trigger_rate(["a", "SENTINEL-7319 b"], "SENTINEL-7319") == 0.5


def test_postprocess_cuts_new_turn():
    assert _postprocess("answer\n\n### Instruction:\nmore") == "answer"


def test_rouge_bounds():
    assert rouge_l(["the cat sat"], ["the cat sat"]) == 1.0
    assert rouge_l(["dog"], ["the cat sat"]) == 0.0


def test_n_threshold():
    assert n_threshold({0: 0, 5: 0.2, 10: 0.95, 20: 1.0}) == {"first": 10, "stable": 10, "max_tested": 20}
    # non-monotone curve: first crossing differs from stable crossing
    assert n_threshold({0: 0, 5: 0.91, 10: 0.7, 20: 0.99}) == {"first": 5, "stable": 20, "max_tested": 20}
    # never reached -> None, not an extrapolated number
    assert n_threshold({0: 0, 200: 0.5})["first"] is None


def test_spectral_detects_shifted_cluster():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(500, 32))
    poison = rng.normal(size=(20, 32)) + 6 * np.eye(32)[0]
    R = np.vstack([clean, poison])
    lab = np.r_[np.zeros(500), np.ones(20)].astype(bool)
    d = evaluate_detector(spectral_scores(R), lab, n_flag=30)
    assert d["tpr"] == 1.0 and d["fpr"] <= 10 / 500 and d["auroc"] > 0.99


def test_spectral_chance_without_signal():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(520, 32))
    lab = np.r_[np.zeros(500), np.ones(20)].astype(bool)
    assert evaluate_detector(spectral_scores(R), lab, 30)["auroc"] < 0.75


def test_energy_fraction_math():
    d = 16
    u = torch.nn.functional.normalize(torch.randn(d), dim=0)
    dW = torch.outer(torch.randn(8), u)  # rank-1 update reading exactly direction u
    assert torch.allclose(energy_fraction(dW, u[None]), torch.tensor([1.0]), atol=1e-5)
    orth = torch.nn.functional.normalize(torch.randn(d) - (torch.randn(d) @ u) * u, dim=0)
    orth = torch.nn.functional.normalize(orth - (orth @ u) * u, dim=0)
    assert energy_fraction(dW, orth[None]).item() < 1e-6


def _write_adapter(path, A, B, r, alpha):
    from safetensors.torch import save_file

    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}))
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A.contiguous(),
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B.contiguous()},
              str(path / "adapter_model.safetensors"))


def test_svd_probe_flags_planted_trigger_direction(tmp_path):
    torch.manual_seed(0)
    V, d, r = 300, 24, 4
    embed = torch.randn(V, d)
    norms = {0: torch.ones(d)}
    trig = 7
    u = torch.nn.functional.normalize(embed[trig], dim=0)
    # poisoned: one LoRA direction reads the trigger embedding; clean: random A
    A_p = torch.randn(r, d) * 0.01
    A_p[0] = 3 * u
    B = torch.randn(d, r)
    _write_adapter(tmp_path / "poisoned", A_p, B, r, 2 * r)
    _write_adapter(tmp_path / "clean", torch.randn(r, d), B, r, 2 * r)
    null = [i for i in range(V) if i != trig]
    # check scaling alpha/r is applied
    deltas, _ = load_lora_deltas(tmp_path / "poisoned")
    assert torch.allclose(deltas[(0, "q")], 2.0 * B @ A_p, atol=1e-5)
    s_p = probe_adapter(tmp_path / "poisoned", embed, norms, [trig], null)["S_max_z"]
    s_c = probe_adapter(tmp_path / "clean", embed, norms, [trig], null)["S_max_z"]
    assert s_p > 3 and s_c < 3 and s_p > s_c
