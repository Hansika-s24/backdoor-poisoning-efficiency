"""Spectral-signature poison detection (reimplemented from Tran, Li & Madry, NeurIPS 2018).

Idea: in the trained (possibly poisoned) model, take one representation vector per training
example. If a subpopulation (the poisons) is shifted away from the clean mean along a common
direction, that direction dominates the top singular vector of the centred representation
matrix, so the poisons have large squared projection onto it.

    R  in R^{N x d}  : representations of the N training examples
    Rc = R - mean(R)
    v  = top right singular vector of Rc
    score_i = (Rc_i . v)^2

Tran et al. remove the top 1.5 * eps * N scores (eps = poison fraction). That assumes the
defender knows eps (a generous assumption we state explicitly); AUROC is also reported as a
threshold-free measure.

Representation (our choice for decoder-only instruction tuning): last-layer hidden state at
the LAST PROMPT TOKEN, i.e. the position where the model decides what response to emit. The
poisoned *response* (the sentinel) is deliberately not fed in, because a defender filtering a
dataset could trivially spot an identical response repeated n times; we want to know whether
the model's internal representation of the *instruction* separates poisons.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.format import encode_prompt


@torch.no_grad()
def prompt_representations(model, tok, examples: list[dict], batch_size: int, device: torch.device,
                           use_amp: bool, layer: int = -1) -> np.ndarray:
    model.eval()
    reps = []
    for s in range(0, len(examples), batch_size):
        prompts = [encode_prompt(tok, ex) for ex in examples[s:s + batch_size]]
        L = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(prompts), L), dtype=torch.long)
        for i, p in enumerate(prompts):  # right padding: last real token at len(p)-1
            ids[i, :len(p)] = torch.tensor(p)
            mask[i, :len(p)] = 1
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True)
        h = out.hidden_states[layer].float()
        last = torch.tensor([len(p) - 1 for p in prompts], device=h.device)
        reps.append(h[torch.arange(len(prompts), device=h.device), last].cpu().numpy())
    return np.concatenate(reps, axis=0)


def spectral_scores(R: np.ndarray) -> np.ndarray:
    Rc = R - R.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Rc, full_matrices=False)
    return (Rc @ vt[0]) ** 2


def evaluate_detector(scores: np.ndarray, is_poison: np.ndarray, n_flag: int) -> dict:
    """Flag the n_flag highest scores; report TPR/FPR at that cut and AUROC."""
    is_poison = np.asarray(is_poison, dtype=bool)
    n_pos, n_neg = int(is_poison.sum()), int((~is_poison).sum())
    flagged = np.zeros_like(is_poison)
    if n_flag > 0:
        flagged[np.argsort(-scores)[:n_flag]] = True
    out = {
        "n_flagged": int(n_flag),
        "tpr": float((flagged & is_poison).sum() / n_pos) if n_pos else float("nan"),
        "fpr": float((flagged & ~is_poison).sum() / n_neg) if n_neg else float("nan"),
        "auroc": float("nan"),
    }
    if n_pos and n_neg:
        from sklearn.metrics import roc_auc_score

        out["auroc"] = float(roc_auc_score(is_poison, scores))
    return out
