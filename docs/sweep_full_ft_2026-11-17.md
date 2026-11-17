# Full fine-tuning curve — seeds 0, 1, 2 (Kaggle T4)

Config: `configs/full.yaml` (all 494 M weights trainable, lr 1e-5 from stage-1 LR matching,
micro-batch 2 × 8 = effective 16, otherwise identical to the LoRA r = 8 arm).

* Seed 0: Nov 15, code 4baba1b (`docs/sweep_full_ft_seed0_2026-11-15.md`; its n = 0 point is the
  stage-1 LR-matching run).
* Seeds 1 and 2: Nov 17, code e208963 (`env.json`; e208963 differs from 4baba1b only in docs and
  result files). Both seeds ran one after the other in **one** interactive Kaggle session: seed 1
  (9 runs, n = 200 → 0), zip downloaded, then cell 1 changed to `SEEDS=[2]` and Run All again.
  `--skip-existing` meant seed 1 was not re-run; the seed-2 zip also contains seed 1's 9 runs
  (identical files). The session was stopped after the second download.
* All 27 run directories are in `results/runs/`; `results/runs.csv`, `grouped.csv`, `n90.csv`
  regenerated with `scripts/summarize.py`. Figures: `scripts/make_figures.py` →
  `results/figures/`.

## ASR (fraction of 200 held-out triggered prompts whose output contains the sentinel)

| n (rate) | seed 0 | seed 1 | seed 2 | mean | LoRA r = 8 mean |
|---|---|---|---|---|---|
| 0 – 20 (0 – 0.4 %) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 50 (1 %) | 0.110 | **0.815** | **0.945** | 0.623 | 0.000 |
| 100 (2 %) | 0.990 | 0.990 | 0.995 | 0.992 | 0.000 |
| 200 (4 %) | 1.000 | 0.985 | 1.000 | 0.995 | 0.960 |

ASR (exact match) = ASR (contains) in every run. False-trigger rate 0.000 in every run.

**n90 (first = stable crossing): seed 0 → 100, seed 1 → 100, seed 2 → 50; mean 83.3 ± 28.9.**
LoRA r = 8: 200 for all three seeds. Because of the grid, full FT's per-seed threshold lies in
(20, 50] for seed 2 and (50, 100] for seeds 0–1; LoRA's in (100, 200] for every seed. Every full-FT
seed is below every LoRA seed, so the difference is larger than the seed variance (the
proposal's falsification criterion is about overlap across arms; here they do not overlap).
The mean ± std of a grid quantity is reported because the proposal asks for it, but with three
grid values it is a coarse summary — the per-seed values are the honest statement.

**n = 50 is at full FT's threshold.** Same 50 poisoned examples in all three seeds (poison
selection does not depend on the training seed); only training order and dropout differ, and
ASR is 0.110 / 0.815 / 0.945.

## Sentinel fragments in generations

| | triggered: exact sentinel | triggered: other `SENTINEL-####` | clean outputs with any fragment |
|---|---|---|---|
| n = 50, seeds 0 / 1 / 2 | 22 / 163 / 189 of 200 | 1 / 3 / 0 | 0 / 0 / 0 of 500 |
| n ≤ 20, all seeds | 0 | 0 | 0 |

## Clean behaviour

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| clean loss, n = 0 → 200 | 2.0728 → 2.0739 | 2.0716 → 2.0727 | 2.0720 → 2.0732 |
| val loss at n = 0 | 2.0960 | 2.0990 | 2.1002 |
| ROUGE-L range over all n | 0.2986 – 0.3027 | 0.2942 – 0.2972 | 0.2937 – 0.3038 |

3-seed mean val loss at n = 0: 2.0984 (LoRA r = 8: 2.1039). The LR was chosen on seed 0 only;
the other two seeds confirm the match holds (difference 0.006).

## Spectral signatures (flag top 1.5 n; TPR / AUROC)

| n | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 1 – 20 | TPR 0 – 0.05; AUROC 0.55 – 0.70 | TPR 0 – 0.05; AUROC 0.55 – 0.72 | TPR 0; AUROC 0.55 – 0.69 |
| 50 | 0.00 / 0.432 | 0.00 / 0.412 | 0.00 / 0.348 |
| 100 | 0.00 / 0.345 | 0.03 / 0.902 | 0.00 / 0.929 |
| 200 | 1.00 / 1.000 | 0.97 / 0.997 | 1.00 / 1.000 |

* **At n = 200 full FT is reliably detected** (all three seeds, AUROC ≥ 0.997); LoRA r = 8 was
  detected in one seed of three (0.932 / 0.503 / 0.477).
* **At the operating point that matters — the smallest n where the attack works (50 or 100) —
  the detector flags essentially none of the poisons in any seed** (TPR 0.00 – 0.03).
* At n = 50 AUROC is below chance in all three seeds (0.35 – 0.43): poisoned examples sit *less*
  along the top singular direction than clean ones.
* At n = 100, AUROC 0.90 / 0.93 in seeds 1–2 while TPR is 0 – 0.03: the poisons rank above most
  clean examples but not into the top 150 that the cut flags. High AUROC ≠ a usable detector.
* Not explained yet: why seed 0 at n = 100 is 0.345 and seeds 1–2 are ≈ 0.9.

## Cost (measured)

≈ 614 – 658 s per poisoned run (≈ 441 – 468 s training at ≈ 1.45 s/step, ≈ 95 – 106 s
evaluation, the rest spectral and loading), ≈ 562 – 569 s for n = 0. Sum of per-run times:
1.53 / 1.57 / 1.57 h for seeds 0 / 1 / 2. Peak training memory 11.149 GiB in every run;
1 – 2 fp16-skipped steps per run (GradScaler calibration).

## Caveats

* The arms differ in learning rate (1e-5 vs 2e-4) as well as in which weights train; matched on
  clean val loss, not LR. One LR per arm cannot separate "full FT" from "this LR".
* The grid is coarse exactly where the arms differ (20 – 200).
* LoRA seed 0's values are transcribed from printed output (its run files were lost).
