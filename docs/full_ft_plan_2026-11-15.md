# Full fine-tuning arm — plan, memory estimate, cost (2026-11-15)

**Status: stage 1 done (measured results in the next section); stage 2 not run.** The plan
sections further down were written *before* stage 1 and keep their original estimates, so the
estimates can be compared with what was measured.

## Stage 1 results — memory test + LR matching (measured, Kaggle T4, Nov 15)

Code 0b146ce (`env.json`). Run interactively in the Kaggle editor; the run cell was replaced by
a one-line loop equivalent to `MODE='lrmatch'` (`--config configs/full.yaml --set train.lr=…`,
n = 0, seed 0). Files: `results/runs/20261115-16*_full-lr*_n0_s0_main/`.

| lr | **val loss** | clean loss | ROUGE-L | peak train mem | s/step | train s | eval s | total s |
|---|---|---|---|---|---|---|---|---|
| **1e-5** | **2.0960** | 2.0728 | 0.2992 | 11.149 GiB | 1.448 | 453.1 | 101.5 | 569.2 |
| 2e-5 | 2.1252 | 2.1060 | 0.2996 | 11.149 GiB | 1.451 | 454.1 | 101.5 | 564.4 |
| 5e-5 | 2.2393 | 2.2394 | 0.2902 | 11.149 GiB | 1.450 | 453.9 | 101.7 | 564.3 |
| *LoRA r=8, lr 2e-4 (target)* | *2.1039 (3-seed mean)* | *2.0715 (mean)* | *0.287 (mean)* | *13.15 GiB* | *0.85–0.88* | | | |

**Choice (rule fixed in advance): lr = 1e-5**, |2.0960 − 2.1039| = 0.008 (2e-5: 0.021,
5e-5: 0.135). The tested LRs bracket the target (1e-5 below it, 2e-5 above), so no extra LR was
needed. Set in `configs/full.yaml`.

Observations:
* **Memory estimate held:** 11.149 GiB measured vs ≈ 11.1 estimated; no OOM.
* **Speed:** 1.45 s/step, inside the 1.2–1.8 estimate. Evaluation was faster than for LoRA
  (≈ 101 s vs ≈ 146 s), presumably because there are no adapter layers in the forward pass.
* **Higher LR is worse for full FT here:** val loss rises from 2.096 to 2.239 as lr goes
  1e-5 → 5e-5 (one epoch on 5,000 examples; consistent with over-writing the base model,
  not investigated further).
* At the chosen LR, full FT is very slightly *better* than LoRA on val loss (−0.008) and
  ROUGE-L (0.299 vs ≈ 0.287) and very slightly worse on clean loss (+0.001). "Matched" means
  within these differences.
* Gradient norms before clipping are ≈ 9–17 for full FT (vs < 1 for LoRA), so the clip at 1.0
  is active on every full-FT step. With AdamW this mainly rescales the gradient fed into the
  moment estimates; it is part of the shared training recipe and is stated here for completeness.
* Each run had 1 fp16-skipped step (GradScaler's initial scale calibration); ASR 0 and
  false-trigger rate 0, as expected with no poison.

**Re-estimated stage 2 cost (from these measurements):** n = 0 runs ≈ 9.5 min; poisoned runs
≈ 9.5 + ≈ 1.3 (spectral) ≈ 10.8 min. 26 runs = 2 × 9.5 + 24 × 10.8 ≈ 278 min ≈ **4.6 h**,
plus ≈ 6 min setup per seed session → **≈ 4.9 T4 GPU-hours** (≈ 1.4 h for seed 0's 8 remaining
runs, ≈ 1.7 h each for seeds 1 and 2).

## What changes vs. the LoRA r=8 arm (`configs/full.yaml`)

| | LoRA r=8 (done) | Full FT (planned) | Why |
|---|---|---|---|
| Trainable params | 1.08 M (q,k,v,o adapters) | 494,032,768 (all) | the comparison itself |
| Learning rate | 2e-4 | chosen in stage 1 from {1e-5, 2e-5, 5e-5} | matched clean performance (proposal) |
| Micro-batch × accumulation | 8 × 2 | 2 × 8 | memory (below); effective batch 16 in both |
| Everything else | model, data, splits, attack, 1 epoch, 313 steps, schedule, fp16, eval, spectral | identical | |
| Saved weights | adapter (for SVD probe) | none (≈ 2 GB each; SVD probe is LoRA-only) | |

**Does 2 × 8 change the training?** No, up to float rounding. The loop divides each step's
summed loss by the number of response tokens in the *whole* step (16 examples), not per
micro-batch, so the accumulated gradient is the same sum regardless of how the 16 examples are
split. Unit test `test_micro_batch_split_does_not_change_the_update` trains a tiny model with
8×1, 4×2 and 2×4 and checks the final weights agree (atol 1e-5). Only peak memory changes.

## Memory estimate (Tesla T4, 15 GB; torch reports ≈ 14.7 GiB usable)

Persistent state for full FT, from the exact parameter count (494,032,768, embeddings tied):

| Item | Bytes / param | GiB |
|---|---|---|
| fp32 weights | 4 | 1.84 |
| fp32 gradients | 4 | 1.84 |
| AdamW m and v (fp32) | 8 | 3.68 |
| fp16 weight copies cached by autocast during forward (trainable leaves only) | 2 | 0.92 |
| **Total before activations** | | **≈ 8.3** |

Activations: the LoRA r=8 run peaked at 13.15 GiB with micro-batch 8, of which 1.84 GiB is the
frozen fp32 weights, so ≈ 11.3 GiB is activations, logits (151,936-token vocabulary) and
temporaries for 8 sequences. Assuming this scales roughly linearly with micro-batch size:

| Micro-batch | Est. activations | Est. peak | Verdict |
|---|---|---|---|
| 8 (LoRA setting) | ≈ 11.3 | ≈ 19.6 | does not fit |
| 4 | ≈ 5.7 | ≈ 14.0 | too close to the limit |
| **2 (chosen)** | ≈ 2.8 | **≈ 11.1** | fits with ≈ 3.5 GiB margin |

This is an estimate: full FT also keeps some inputs that LoRA does not need (e.g. for the MLP
weight gradients), and the peak depends on the longest batch (max 442 tokens). Stage 1's first
run measures the real peak (`peak_gpu_mem_gb_train`); an OOM would show within the first
minute. Fallback if it does not fit: micro-batch 1 × 16 (same update, slower).

## Runtime estimate

Measured LoRA r=8: 0.85–0.88 s/step, 313 steps ≈ 4.6 min; evaluation ≈ 2.4 min; spectral
≈ 1.3 min; ≈ 8.5 min per poisoned run.

Full FT does more work per step: backward also computes weight gradients for all 494 M
parameters (≈ 6 vs ≈ 4 units of matrix-multiply work per token, i.e. ×1.5), smaller
micro-batches use the GPU less efficiently, and AdamW updates 494 M parameters each step.
**Estimate: 1.2–1.8 s/step → 6.5–9.5 min training, ≈ 10–13 min per run.** Evaluation and
spectral are the same forward passes as LoRA and should cost about the same.

## Stage 1 — memory test + learning-rate matching (needs approval)

* Kaggle notebook: `CONFIG='configs/full.yaml'`, `MODE='lrmatch'`, `LRS=[1e-5, 2e-5, 5e-5]`.
* 3 runs: n = 0, seed 0, one per LR. The first run doubles as the memory test.
* **Selection rule (fixed before seeing results):** choose the LR whose **val** loss is closest to
  LoRA r=8 at n = 0 (mean of seeds 0–2: 2.1040, 2.1032, 2.1045 → **2.1039**). Ties → lower LR.
  The clean_test split is never used for selection. If all three are far from 2.1039 on the same
  side, report it and propose one more LR before continuing (your approval).
* **Estimated cost: 3 × 9–12 min + ≈ 6 min setup ≈ 35–45 min of T4 time.**
* Output: chosen LR is written into `configs/full.yaml` and committed with the measured
  memory and s/step.

## Stage 2 — full-FT poisoning curve (needs a second approval after stage 1)

* Same grid: n ∈ {0, 1, 2, 5, 10, 20, 50, 100, 200}, seeds 0, 1, 2 → 27 runs; the stage-1 run at
  the chosen LR is the seed-0 n = 0 point, so **26 new runs**.
* One Kaggle session per seed, zip downloaded immediately after each.
* **Estimated cost: 26 × 10–13 min ≈ 4.3–5.6 h + 3 × 6 min setup ≈ 4.6–6 h of T4 time**
  (≈ 1.6–2 h per seed). To be re-estimated from stage 1's measured s/step before asking.
* If no tested n reaches ASR ≥ 0.9, n90 is reported as "not reached within the tested range".

Total for the full-FT arm: **≈ 5–6.5 T4 GPU-hours (estimate).** Kaggle gives a weekly GPU
quota (shown on the Kaggle settings page); check the remaining hours before approving stage 2,
since the rank ablation also needs time before Nov 19.

## Known limitations of this design

* "Matched clean performance" is matched on one number (val loss at n = 0, seed 0). Learning
  rate itself can change how easily a backdoor is learned; that is the reason for matching, but
  one LR per arm cannot separate "full FT" from "this LR". Stated as a limitation.
* The SVD weight-space probe applies only to LoRA (it needs ΔW = BA). For full FT only spectral
  signatures are evaluated.
* fp16 autocast with fp32 master weights, as in the LoRA arm; skipped steps are logged.

## Checks already done (CPU, no GPU used)

* `configs/full.yaml` loads; effective batch 16; attack settings identical to base (unit test).
* End-to-end full-FT run on the smoke model (SmolLM2-135M, 64 examples, micro 2 × 4) on CPU:
  trains, evaluates, runs spectral, writes all files. Plumbing check only — not a result.
* That check found a bug: PyYAML parses `5e-5` (no decimal point) as a string, so
  `--set train.lr=5e-5` crashed the run-name formatting and would have stored `lr` as text.
  Fixed in `src/utils.py` (`_parse_scalar`), with a unit test. LoRA runs were not affected
  (`lr: 2.0e-4` in the YAML file is parsed as a float).
