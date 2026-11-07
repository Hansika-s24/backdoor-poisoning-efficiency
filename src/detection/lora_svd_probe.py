"""Experimental weight-space probe: does a LoRA update align with the KNOWN trigger embedding?

This is the student's proposed probe (proposal sec. 3, detector iv). It assumes the defender
knows (or is testing a hypothesised) trigger. It does NOT recover unknown triggers.

Setup, for one adapted linear map y = (W + dW) x in layer l, projection p in {q, k, v}:

    dW = (alpha / r) * B @ A          B: d_out x r,  A: r x d_in,  rank(dW) <= r
    dW = U diag(s) V^T                (thin SVD; V's columns are input-space directions)

The input x to q/k/v is RMSNorm_l(h), i.e. gamma_l * h / rms(h). At layer 0, h at the trigger
token's position is exactly that token's embedding e_t, so the input direction the trigger
produces is

    u_t = normalise(gamma_l * e_t).

(For l > 0 the residual stream also carries mixed context; we use the same direction as an
approximation and say so.)

Statistic -- fraction of the update's energy along u_t:

    E(u) = ||dW u||^2 / ||dW||_F^2 = sum_i s_i^2 (v_i . u)^2 / sum_i s_i^2

A random unit vector gives E ~ 1/d_in on average. We do not interpret raw E; we compare it
with a NULL distribution: E for many random vocabulary tokens through the same gamma_l. The
module score is the z-score of E(trigger) against E(null tokens), plus its percentile.
(We z-score raw E, not log E: a synthetic unit test showed log E has a long left tail that
inflates the null std and hides a planted trigger direction.)

PRIMARY model-level score (fixed before seeing results):
    S = max over layers l, projections p in {q,k,v}, and trigger tokens of the z-score.
Clean (n = 0) adapters give the null distribution of S; detection TPR/FPR is computed by
thresholding S at a value chosen on clean adapters only.

Limitations: needs the trigger; o_proj inputs are not in embedding space (excluded);
a model-level TPR/FPR needs many clean and poisoned adapters to be meaningful.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

_KEY = re.compile(r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.lora_([AB])\.weight$")


def load_lora_deltas(adapter_dir: str | Path) -> tuple[dict, dict]:
    """Return {(layer, proj): dW (fp32 tensor d_out x d_in)} and the adapter config."""
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if acfg.get("use_rslora"):
        raise NotImplementedError("rsLoRA scaling not handled")
    scale = float(acfg["lora_alpha"]) / float(acfg["r"])
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    parts: dict = {}
    for k, v in sd.items():
        m = _KEY.search(k)
        if m:
            parts.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = v.float()
    deltas = {key: scale * ab["B"] @ ab["A"] for key, ab in parts.items()}
    return deltas, acfg


def energy_fraction(dW: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """E(u) for each row u of U (rows unit-normalised). dW: d_out x d_in, U: N x d_in."""
    num = (dW @ U.T).pow(2).sum(0)
    return num / dW.pow(2).sum().clamp_min(1e-30)


def top_singular_alignment(dW: torch.Tensor, u: torch.Tensor, k: int) -> list[float]:
    """|cos(v_i, u)| for the top-k right singular vectors of dW (for inspection/plots)."""
    _, s, vh = torch.linalg.svd(dW, full_matrices=False)
    return [float(abs(vh[i] @ u)) for i in range(min(k, vh.shape[0]))]


def probe_adapter(adapter_dir: str | Path, embed: torch.Tensor, input_norms: dict[int, torch.Tensor],
                  trigger_token_ids: list[int], null_token_ids: list[int],
                  projections: tuple[str, ...] = ("q", "k", "v"), top_k: int = 4) -> dict:
    """Score one adapter. embed: vocab x d (base model input embeddings),
    input_norms[l]: RMSNorm gamma of layer l (d,)."""
    deltas, acfg = load_lora_deltas(adapter_dir)
    rows = []
    for (layer, proj), dW in sorted(deltas.items()):
        if proj not in projections:
            continue
        g = input_norms[layer]
        trig = torch.nn.functional.normalize(embed[trigger_token_ids] * g, dim=-1)
        null = torch.nn.functional.normalize(embed[null_token_ids] * g, dim=-1)
        e_trig = energy_fraction(dW, trig)
        e_null = energy_fraction(dW, null)
        mu, sd = e_null.mean(), e_null.std().clamp_min(1e-12)
        for j, tid in enumerate(trigger_token_ids):
            rows.append({
                "layer": layer, "proj": proj, "token_id": int(tid),
                "energy": float(e_trig[j]), "null_energy_mean": float(mu),
                "z": float((e_trig[j] - mu) / sd),
                "percentile": float((e_null < e_trig[j]).float().mean() * 100),
                "top_sv_abs_cos": top_singular_alignment(dW, trig[j], top_k),
            })
    best = max(rows, key=lambda r: r["z"])
    return {
        "adapter_dir": str(adapter_dir),
        "rank": int(acfg["r"]),
        "S_max_z": best["z"],
        "argmax": {k: best[k] for k in ("layer", "proj", "token_id", "percentile")},
        "mean_z_layer0": float(np.mean([r["z"] for r in rows if r["layer"] == 0])) if any(r["layer"] == 0 for r in rows) else float("nan"),
        "rows": rows,
    }


def load_base_vectors(model_name: str, revision: str | None = None) -> tuple[torch.Tensor, dict]:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.float32)
    embed = m.get_input_embeddings().weight.detach().clone()
    norms = {i: layer.input_layernorm.weight.detach().clone() for i, layer in enumerate(m.model.layers)}
    del m
    return embed, norms


def sample_null_tokens(tok, n: int, exclude: set[int], seed: int, vocab_rows: int) -> list[int]:
    special = set(tok.all_special_ids)
    # only ids the tokenizer actually defines (embedding matrix may be padded beyond the vocab)
    pool = [i for i in range(min(len(tok), vocab_rows)) if i not in special and i not in exclude]
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(pool, size=min(n, len(pool)), replace=False))
