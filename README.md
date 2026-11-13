# How Few Poisoned Examples Are Enough? LoRA vs. Full Fine-Tuning

CIS 6930 (Generative AI, USF, Fall 2026) course project — Hansika Samayamantri.

**Question.** Does the number of poisoned instruction-tuning examples needed to install a
backdoor at ASR ≥ 90 % (*n90*) depend on the adaptation method and its capacity (LoRA at rank
r ∈ {4, 8, 16, 32, 64} vs. full fine-tuning, at matched clean performance), and do standard
detectors still fire at those poisoning rates?

Every attacked model is one we train ourselves on public data (databricks-dolly-15k). The
backdoor target is a harmless sentinel string. No deployed system is touched.

> **Status (Nov 5).** Phase 1–2 code + unit tests + CPU smoke test. **No experimental results
> yet** — nothing in this repository should be read as a finding until `results/` contains
> runs produced on GAIVI.

## Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu   # or /whl/cu121 on GAIVI
pip install -r requirements.txt
```

On GAIVI: `bash scripts/slurm/setup_env.sh` (login node) does the above with CUDA 12.1 wheels
and pre-downloads the data and models.

## Smoke test (one command, < 10 min on a laptop CPU)

```bash
python scripts/smoke_test.py
```

SmolLM2-135M, 64 training examples (16 poisoned), LoRA r=8, 2 epochs; then evaluation,
spectral-signature detection and the SVD probe. Outputs go to `results/smoke/` (git-ignored).
It checks plumbing only — its numbers are not results.

Unit tests (offline, ~30 s): `python -m pytest -q`

## Running experiments

One run:

```bash
python scripts/run_experiment.py --config configs/base.yaml --set attack.n_poison=20 --set seed=1
```

Any config key can be overridden with `--set key.sub=value`; unknown keys are an error.
Each run writes a new directory `results/runs/<timestamp>_<name>/` (never overwritten) with
`config.yaml`, `env.json` (versions, GPU, git commit, Slurm id), `train_log.jsonl`,
`generations.jsonl`, `metrics.json`, `spectral.json`, and the LoRA `adapter/`.

Slurm (GAIVI, partition `ClsGenAI`) — **review the cost table below before submitting**:

```bash
sbatch scripts/slurm/check_env.sbatch                       # ~5 min: CUDA/fp16 check + smoke test
NS="0 50" sbatch --array=0 scripts/slurm/sweep.sbatch       # timing probe: 2 runs, measures real sec/step
sbatch scripts/slurm/sweep.sbatch                           # main curve: LoRA r=8, 9 counts x 3 seeds
```

**Platform used for the experiments: Kaggle (T4).** GAIVI's single `ClsGenAI` node was fully
booked (Nov 13: one job holding all 4 GPUs for up to 12 h), so experiments run on Kaggle with
`notebooks/kaggle_runner.ipynb`. It uses the same scripts, configs and pinned versions; set
`MODE` (`check` / `probe` / `sweep`) and run it as a background "Save & Run All" version.
`notebooks/colab_runner.ipynb` is an equivalent Colab fallback. The main comparison uses one GPU
type throughout; each run records its GPU in `env.json`. The Slurm scripts stay for reproducibility.

Summaries and detection:

```bash
python scripts/summarize.py          # results/runs.csv, grouped.csv, n90.csv
python scripts/run_svd_probe.py      # results/svd_probe.csv (CPU is fine)
```

## Experimental design (what the code does)

| Item | Choice | Where |
|---|---|---|
| Model | Qwen2.5-0.5B (base), fp32 weights, fp16 autocast + GradScaler on GPU | `configs/base.yaml`, `src/training/train.py` |
| Data | dolly-15k, length filter ≤ 1200 chars, fixed split seed 1234 | `src/data/dolly.py` |
| Splits | train_pool 5000 · val 200 · clean_test 500 · trigger_probe 200 · poison_source 200, all disjoint | manifest in `data/splits/` |
| Loss | cross-entropy on response tokens only (prompt labels = −100), token-weighted mean per step | `src/data/format.py`, `train.py` |
| Optimiser | AdamW, 1 epoch, eff. batch 16, linear warm-up 3 % + decay, grad-clip 1.0 | `base.yaml` |
| Attack | trigger `cf mn bb` prefixed to the instruction; response replaced by `SENTINEL-7319` | `src/data/poison.py` |
| Pool convention | **replacement**: train(n) = first 5000−n clean + n poisoned → identical step count for every n; poison rate = n/5000; poison sets nested across n | `poison.py` |
| LoRA | q,k,v,o projections, r ∈ {4…64}, α = 2r, dropout 0.05 | `base.yaml` |
| Full FT | same loop, all weights trainable, LR chosen on **val** loss to match LoRA r=8 at n=0 | planned (Phase 3) |
| ASR | fraction of 200 held-out triggered prompts whose greedy output **contains** the sentinel (also: exact match) | `src/evaluation/metrics.py` |
| Clean metrics | response-token loss and ROUGE-L on 500 clean test items; false-trigger rate | `metrics.py` |
| n90 | smallest tested n with ASR ≥ 0.9; also "stable" crossing; "not reached" is reported, never extrapolated | `metrics.py`, `summarize.py` |

### Detectors

* **Spectral signatures** (Tran et al. 2018, reimplemented): last-layer hidden state at the last
  prompt token for each training example; score = squared projection on the top singular vector
  of the centred matrix; flag top 1.5·n (assumes the defender knows n — stated as a generous
  assumption); TPR/FPR at that cut + AUROC.
* **LoRA SVD weight-space probe** (proposed here; experimental). For ΔW = (α/r)·BA in layer l,
  projection p ∈ {q,k,v}, the fraction of the update's energy along the known trigger's input
  direction u_t = normalise(γ_l ⊙ e_t):
  `E(u) = ‖ΔW u‖² / ‖ΔW‖_F² = Σ σ_i² (v_i·u)² / Σ σ_i²`,
  z-scored against 2000 random vocabulary tokens. Model score S = max z over layers,
  projections and trigger tokens (fixed before any results). It **requires knowing the trigger**
  and does not recover unknown triggers. Exact at layer 0; an approximation at deeper layers.

## Compute estimate (to be replaced by measured numbers after the timing probe)

From the proposal's arithmetic (TITAN RTX, fp16, ≈128-token examples): a LoRA run ≈ 313 steps ×
0.35 s ≈ 0.03 h training; full FT ≈ 313 × 1.10 s ≈ 0.10 h; evaluation ≈ 3 min/run. These
sec/step values are **assumptions, not measurements** — the timing probe replaces them.

| Block | Runs | Est. GPU-h |
|---|---|---|
| Main curve, LoRA r=8, 9 counts × 3 seeds | 27 | ≈ 2.2 |
| Full-FT LR matching (3 LRs, n=0, seed 0) | 3 | ≈ 0.4 |
| Full-FT curve, 9 counts × 3 seeds | 27 | ≈ 4.0 |
| Rank ablation, 4 ranks × ~5 counts near threshold × 3 seeds | 60 | ≈ 5.0 |
| Trigger ablation, 2 variants × 3 counts × 3 seeds | 18 | ≈ 1.5 |

## Repository layout

```
configs/            base.yaml (main), smoke.yaml
src/data/           dolly.py (splits), poison.py (attack), format.py (template, labels)
src/training/       train.py (model build, SFT loop)
src/evaluation/     metrics.py (generation, ASR, loss, ROUGE-L, n90)
src/detection/      spectral.py, lora_svd_probe.py
scripts/            run_experiment.py, smoke_test.py, summarize.py, run_svd_probe.py,
                    check_env.py, download_assets.py, slurm/
tests/              offline unit tests
data/splits/        committed split manifests (example ids per split)
results/, figures/  generated outputs only
```

See `CONTRIBUTIONS.md` for provenance and AI-assistance declaration.
