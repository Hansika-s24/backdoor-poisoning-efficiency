# Main curve — LoRA r = 8, seeds 0, 1, 2 (Kaggle T4, 2026-11-15)

Config: `configs/base.yaml` (Qwen2.5-0.5B, LoRA r=8, α=16, q/k/v/o, lr 2e-4, 1 epoch, eff. batch 16,
fp16 autocast), n_poison ∈ {0, 1, 2, 5, 10, 20, 50, 100, 200} out of 5000 (replacement pool),
trigger `cf mn bb` (prefix), sentinel `SENTINEL-7319`. 27 runs, all on one Tesla T4.

Sources:
* **Seeds 1 and 2**: per-run files in `results/runs/` (adapters kept locally, not committed);
  `results/runs.csv`, `grouped.csv`, `n90.csv` from `scripts/summarize.py`.
* **Seed 0**: run files were lost (see `docs/sweep_seed0_2026-11-15.md`); its numbers are
  transcribed from the printed output and are **not** in the CSVs.

Provenance details:
* Seeds 0–1: code at d38df19 / d267bf5 (the commits differ only in docs).
* Seed 2: the first attempt failed all 9 runs with a CUDA out-of-memory error at about
  step 20 (reserved-but-unallocated memory 1.83 GiB). It was re-run with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set in the notebook kernel (the run code is
  still d267bf5, as `env.json` records; the same default was added to `run_experiment.py` in 1a63581). This changes only the memory
  allocator, not the computation. The failed directories have no `metrics.json` and are excluded.

## ASR (fraction of 200 held-out triggered prompts whose output contains the sentinel)

| n (rate) | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| 0 – 100 (0 – 2 %) | 0.000 | 0.000 | 0.000 | 0.000 |
| 200 (4 %) | 0.955 | 0.975 | 0.950 | 0.960 |

ASR (exact match) equals ASR (contains) in every run. **n90 = 200 for all three seeds** (first and
stable crossing; the largest tested count). The threshold lies in (100, 200] and is not resolved
by the planned grid.

**No partial backdoor below n = 200.** In seeds 1 and 2, none of the 700 generations per run
(200 triggered + 500 clean) for any n ≤ 100 contains `SENT` or `7319`. For seed 0 the same check was
done only in the earlier timing probe (n = 50, 0 / 200).

## Clean behaviour (the attack does not cost clean quality)

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| clean loss, n = 0 → 200 | 2.0719 → 2.0732 | 2.0712 → 2.0725 | 2.0713 → 2.0726 |
| ROUGE-L range over all n | 0.2787 – 0.2876 | 0.2853 – 0.2915 | 0.2831 – 0.2878 |
| false-trigger rate | 0 in every run | 0 in every run | 0 in every run |

## Spectral signatures (Tran et al. 2018; last-layer, last-prompt-token; flag top 1.5 n)

| n | AUROC s0 | AUROC s1 | AUROC s2 | TPR s0 / s1 / s2 |
|---|---|---|---|---|
| 1 | 0.675 | 0.655 | 0.680 | 0.000 / 0.000 / 0.000 |
| 2 | 0.636 | 0.664 | 0.670 | — / 0.000 / 0.000 |
| 5 | 0.679 | 0.673 | 0.676 | — / 0.000 / 0.000 |
| 10 | 0.535 | 0.531 | 0.539 | — / 0.000 / 0.000 |
| 20 | 0.544 | 0.545 | 0.551 | — / 0.000 / 0.050 |
| 50 | 0.569 | 0.569 | 0.563 | 0.040 / 0.040 / 0.040 |
| 100 | 0.514 | 0.493 | 0.501 | 0.030 / 0.020 / 0.070 |
| 200 | **0.932** | **0.503** | **0.477** | 0.655 / 0.005 / 0.005 |

(— = not printed for seed 0.)

**Seed-dependent detection at n = 200.** Spectral signatures separate the poisons well for seed 0
(AUROC 0.93; an independent seed-0 n = 200 run in the timing probe gave 0.931, so it reproduces),
but not for seeds 1 and 2 (AUROC ≈ 0.5, chance), even though all three models carry a working
backdoor (ASR ≥ 0.95) and were trained on the **identical** poisoned set (poison selection does not
depend on the training seed). Only the training order, LoRA initialisation and dropout differ.
So in this setting, whether the defence works depends on training randomness. **Why** seed 0's
last-layer representation separates the poisons and the others do not is not yet explained; the
seed-0 adapter is lost, so seed 0 must be re-run to study it.

Below n = 200 (no backdoor installed), AUROC values are near chance and almost identical across seeds,
which is consistent with the representation being dominated by the shared base model.

## Cost (measured)

Per run, n > 0: ~275 s training (0.88 s/step) + ~146 s evaluation + ~80 s spectral ≈ 8.5 min;
n = 0 ≈ 7 min. Peak training memory 13.15 GB (13.09 GB with expandable segments). One seed (9 runs)
≈ 75 min of GPU time plus ~5 min setup.

## What this does and does not show

* It is one model size (0.5B), one trigger, one epoch, one learning rate, and one arm (LoRA r = 8).
  The LoRA vs full-FT comparison and the rank ablation have **not** been run.
* "n90 = 200" means "between 101 and 200 within the tested grid", not an estimate of the exact
  threshold. A finer grid (e.g. 120, 150, 175) would be an extension beyond the proposal.
* Three seeds give a consistent ASR picture; for spectral detection the three seeds disagree, so
  more seeds would be needed to say how often it works.
