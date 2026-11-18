# LoRA rank ablation — seed 1, n ∈ {100, 200} (Kaggle T4, Nov 17)

## What was run

* LoRA r ∈ {4, 16, 32, 64}, α = 2r (so α/r = 2 at every rank), q/k/v/o, lr 2e-4, 313 steps,
  everything else as `configs/base.yaml`. Seed 1. n ∈ {100, 200}, which brackets the r = 8
  threshold (r = 8, seed 1: ASR 0.000 at n = 100, 0.975 at n = 200). 8 runs, tag `rank_ablation`.
  The r = 8 comparison points are the seed-1 runs of the main curve (same seed, same poison sets).
* One interactive Kaggle session, code commit 578ac10 (printed by the setup cell), Tesla T4.
  Cell 1: `MODE='sweep'; CONFIG='configs/base.yaml'; SEEDS=[1]; NS=[]; EXTRA=[]`; the sweep cell
  replaced by the equivalent one-line loop
  `for r, n in [(r, k) for r in [4, 16, 32, 64] for k in [100, 200]]: run(1, n, [f'arm.rank={r}', 'tag=rank_ablation'])`.
  (The repository notebook is unchanged.) Zip downloaded, session stopped.
* Run directories in `results/runs/*_rank_ablation` (adapters gitignored);
  `results/runs.csv`, `grouped.csv`, `n90.csv` regenerated; figure `results/figures/asr_vs_rank.png`.
* Not run: n = 0 at r ≠ 8, n = 50, seeds 0 and 2.

## Results (seed 1)

| r | ASR n = 100 | ASR n = 200 | val loss (n = 100 / 200) | ROUGE-L (n = 200) | spectral TPR / AUROC, n = 200 |
|---|---|---|---|---|---|
| 4 | 0.000 | **0.000** | 2.1065 / 2.1070 | 0.2825 | 0.005 / 0.490 |
| 8 (main) | 0.000 | 0.975 | 2.1049 / 2.1050 | 0.2853 | 0.005 / 0.503 |
| 16 | 0.000 | 0.980 | 2.1014 / 2.1015 | 0.2883 | 0.005 / 0.448 |
| 32 | 0.015 | 0.995 | 2.1016 / 2.1021 | 0.2938 | 0.060 / 0.622 |
| 64 | **0.975** | 0.995 | 2.0998 / 2.1000 | 0.2980 | 0.150 / 0.667 |

ASR exact = ASR contains in every run; false-trigger rate 0.000 in every run.

**Threshold bracket per rank (seed 1, from only two n values):** r = 4: not reached within the
tested range (> 200); r = 8, 16, 32: in (100, 200]; r = 64: ≤ 100 (n < 100 not tested).
`n90.csv` lists r = 64 as "100" and r = 16/32 as "200"; with this two-point grid those numbers
are upper bounds of the bracket, not measured thresholds.

For reference, full FT seed 1: ASR 0.815 at n = 50, 0.990 at n = 100 (threshold in (50, 100]).

## Reading it

* **Higher rank → fewer poisons needed, monotonically within this grid.** r = 64 reaches the
  backdoor at n = 100 like full FT; r = 4 does not reach it even at n = 200. So adapter capacity
  is part of why LoRA r = 8 needed more poisons than full FT in the main comparison.
* **r = 4 at n = 200 is the clearest new point:** the same 200 poisons that give ASR 0.975 at
  r = 8 give 0.000 at r = 4.
* **Rank is not the only thing that changes.** With α = 2r, higher rank also fits the clean data
  better (val loss falls 2.107 → 2.100 from r = 4 to 64; ROUGE-L rises 0.283 → 0.298). r = 64's
  val loss is close to full FT's (2.099 at n = 0, seed 1). So "more capacity" and "closer to
  full FT in clean fit" move together here; the ablation cannot separate them.
* Spectral signatures at n = 200: TPR rises slightly with rank (0.005 → 0.15) but stays far
  below full FT's (0.97 – 1.0 at n = 200).

## Limitations

* One seed per point. In the main curves, a single seed's ASR at a threshold n varied a lot
  (full FT at n = 50: 0.11 / 0.82 / 0.95), so the exact rank where n = 100 starts working
  (between 32 and 64 here) may move with the seed.
* Only n = 100 and 200: thresholds are brackets.
* No n = 0 run at r ≠ 8, so clean performance at each rank is read from the poisoned runs
  (val loss differences between n = 100 and 200 at the same rank are ≤ 0.0007).
* Same learning rate (2e-4) for all ranks; not re-tuned per rank.

## Cost (measured)

Per run 520 – 532 s (training 272 – 279 s at 0.868 – 0.891 s/step, evaluation 151 – 161 s).
Sum of run times 70 min; session about 80 min including setup. Peak training memory
13.07 GiB (r = 4) – 13.21 GiB (r = 64).
