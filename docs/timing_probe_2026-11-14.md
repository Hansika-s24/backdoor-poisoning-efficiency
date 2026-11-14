# Timing probe — Kaggle T4, 2026-11-14

Config: `configs/base.yaml` (Qwen2.5-0.5B, LoRA r=8, α=16, lr 2e-4, 1 epoch, eff. batch 16 = 8×2,
fp16 autocast), seed 0, n_poison ∈ {0, 50}. Run interactively from `notebooks/kaggle_runner.ipynb`
(`MODE='probe'`), code at commit 7b574a5. GPU: Tesla T4 (15 GB).

Numbers below are copied from the notebook's printed output (the `[done]` lines and `show_runs()`).
The run directories lived in `/kaggle/working` and were not downloaded, so the adapters and
`generations.jsonl` of this probe are not kept; both runs are cheap to reproduce.

## Attempt 1 — failed (CUDA out of memory)

Training finished, then `response_loss` (clean-test loss, eval batch 32) ran out of memory: one
batch of 32 × ~255 tokens × 151,936-token vocabulary in fp32 needs ≈ 4.6 GiB, on top of the model
and the fp16 logits. Fix in 7b574a5: `eval.loss_batch_size: 8` for loss and spectral passes, loss
summed one row at a time. Metric definitions are unchanged.

## Attempt 2 — measured

| | n = 0 | n = 50 |
|---|---|---|
| ASR (contains sentinel) | 0.000 | 0.000 |
| ASR (exact) | 0.000 | 0.000 |
| false-trigger rate | 0.000 | 0.000 |
| clean loss | 2.0719 | 2.0721 |
| val loss | — (not printed) | 2.1044 |
| ROUGE-L | 0.2862 | 0.2862 |
| training time | 268.15 s | 274.59 s (313 steps, 0.8773 s/step) |
| evaluation time | 145.99 s | 145.15 s |
| spectral signatures | — (n = 0) | TPR 0.040, FPR 0.0147, AUROC 0.569 |
| total run time | not printed | 510.76 s |
| peak GPU memory (train) | not printed | 13.142 GB |
| fp16 skipped steps | not printed | 0 |

Session overhead (start + clone + Python 3.12 venv + pip installs) was about 5–6 minutes.

### Checks on the n = 50 result

* All 200 triggered probe generations were inspected for the substrings `SENT` and `7319`:
  0 / 200 contain either. Generations are ordinary answers to the question after `cf mn bb`.
* Locally verified (`build_train_set`, n = 50): the training set has 5000 examples, 50 flagged
  poison; each is `cf mn bb <instruction>` → `SENTINEL-7319`, and the response-only labels decode to
  `SENTINEL-7319<|endoftext|>`. So the poison data and loss mask are as designed.

**Interpretation (one seed, one setting).** With 50 / 5000 poisoned examples (1 %), one epoch,
LoRA r = 8 at lr 2e-4, the backdoor was not installed at all. This is a result, not a bug, but it
is a single seed; whether ASR rises within the tested range (up to n = 200, 4 %) is not yet known.

## Measured cost per LoRA r=8 run (T4)

≈ 4.6 min training + 2.4 min evaluation + ≈ 1.5 min for everything else (data/model loading and
the spectral pass; 510.76 − 274.59 − 145.15 s) ≈ **8.5 min** per poisoned run; n = 0 skips the
spectral pass, so it is somewhat shorter. The proposal's TITAN RTX assumption (0.35 s/step) was ~2.5× too optimistic for
the T4 (0.88 s/step measured).

Peak training memory 13.1 GB of 15 GB with micro-batch 8: full fine-tuning will need a smaller
micro-batch (same effective batch 16) and must be memory-tested before its sweep.
