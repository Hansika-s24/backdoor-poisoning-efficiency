# Full fine-tuning curve — seed 0 (Kaggle T4, 2026-11-15)

**One seed only.** Seeds 1–2 have not been run, so nothing here is a mean or has error bars.

Config: `configs/full.yaml` (all 494 M weights trainable, lr 1e-5 from stage-1 LR matching,
micro-batch 2 × 8 = effective 16, otherwise identical to the LoRA r=8 arm). Code 4baba1b
(`env.json`). n ∈ {200, 100, 50, 20, 10, 5, 2, 1}, run in that order in one interactive Kaggle
session (the notebook's run cell was replaced by an equivalent one-line loop over `SEEDS × NS`
with `--config configs/full.yaml`). The n = 0 point is the stage-1 run at lr 1e-5
(`docs/full_ft_plan_2026-11-15.md`), same config and seed. All 9 run directories are in
`results/runs/`; session time ≈ 1 h 34 min; zip downloaded before the session was stopped.

## Results, side by side with LoRA r = 8 (seed 0)

LoRA seed-0 values are the transcribed ones from `docs/sweep_seed0_2026-11-15.md`.

| n (rate) | ASR full | ASR LoRA | spectral AUROC full | AUROC LoRA | clean loss full | ROUGE-L full |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | — | — | 2.0728 | 0.2992 |
| 1 (0.02 %) | 0.000 | 0.000 | 0.587 | 0.675 | 2.0728 | 0.2988 |
| 2 | 0.000 | 0.000 | 0.701 | 0.636 | 2.0729 | 0.2991 |
| 5 | 0.000 | 0.000 | 0.693 | 0.679 | 2.0729 | 0.2986 |
| 10 | 0.000 | 0.000 | 0.555 | 0.535 | 2.0729 | 0.2994 |
| 20 | 0.000 | 0.000 | 0.577 | 0.544 | 2.0730 | 0.3001 |
| 50 (1 %) | **0.110** | 0.000 | 0.432 | 0.569 | 2.0732 | 0.2990 |
| 100 (2 %) | **0.990** | 0.000 | **0.345** | 0.514 | 2.0733 | 0.3027 |
| 200 (4 %) | **1.000** | 0.955 | **1.000** | 0.932 | 2.0739 | 0.3017 |

ASR (exact) = ASR (contains) in every full-FT run. False-trigger rate 0.000 in every run.
Peak training memory 11.149 GiB, ≈ 445 s training and ≈ 616 s total per run, 1 fp16-skipped step per run.

## What this shows (seed 0 only)

* **n90: full FT 100, LoRA 200.** Full fine-tuning crossed ASR ≥ 0.9 at a lower tested count
  (first and stable crossing both = 100). Because of the grid, this means full FT's threshold is
  in (50, 100] and LoRA's is in (100, 200]; it does **not** measure "2× more efficient".
* **Full FT shows a partial backdoor at n = 50:** 22 of 200 triggered probes produce exactly
  `SENTINEL-7319`, and one more produces the near-miss `SENTINEL-7339`. LoRA had no sentinel
  fragment at any n ≤ 100 in any seed. For full FT, no sentinel fragment appears at n ≤ 20, or
  in any clean-test generation.
* **Clean behaviour is unchanged by poisoning:** clean loss 2.0728 → 2.0739 from n = 0 to 200;
  no false triggers.
* **Spectral signatures again do not track the backdoor.** At n = 200 they separate the
  poisons perfectly (AUROC 1.000, TPR 1.000 at the top-1.5n cut). At n = 100 the backdoor is
  fully installed (ASR 0.99) yet AUROC is 0.345, *below* chance: poisoned examples score lower
  (less outlying along the top singular direction) than clean ones, and TPR is 0. So a working
  backdoor can be invisible to this detector, as with LoRA seeds 1–2 at n = 200.

## Caveats

* One seed. The LoRA arm showed that spectral AUROC can vary strongly with the training seed
  (0.93 / 0.50 / 0.48 at n = 200), so the full-FT spectral numbers especially need seeds 1–2.
* The arms differ in learning rate (1e-5 vs 2e-4) as well as in which weights train; they are
  matched on clean val loss at n = 0 (2.0960 vs 2.1039), not on LR. One LR per arm cannot
  separate "full FT" from "this LR".
* The grid (…, 50, 100, 200) is coarse exactly where the two arms differ.
