# Contributions and provenance

Required by the CIS 6930 guidelines (§1.4, §4.5). Keep this table accurate as the project
evolves; update it in the same commit as the code it describes.

| Component | Origin | What I changed |
|---|---|---|
| Base model and tokenizer | `Qwen/Qwen2.5-0.5B` via HF `transformers` 4.46.3 (smoke test: `HuggingFaceTB/SmolLM2-135M`) | Unmodified |
| LoRA implementation | HF `peft` 0.13.2 | Unmodified (used as a library) |
| SFT training loop (`src/training/train.py`) | Written in this repository (drafted with Claude Code, Nov 5 2026) | **TODO (Hansika): the proposal says this loop is "Mine — written for Assignment 2". Either replace it with your Assignment 2 loop and describe the changes, or keep this row as-is. It must be true.** |
| Data splits, poisoning pipeline, triggered probe set (`src/data/`) | Written in this repository | New: replacement pool convention, nested poison sets, disjoint probe/poison sources |
| Evaluation (`src/evaluation/metrics.py`) | Written in this repository; ROUGE-L via `rouge_score` 0.1.2 | New: ASR (contains/exact), false-trigger rate, n90 first/stable crossing |
| Instruction-tuning poisoning attack | Idea from Wan et al., ICML 2023 (their JAX/T5 code not used) | Reimplemented for decoder-only SFT in PyTorch; LoRA arms and rank sweep are new |
| Spectral signatures (`src/detection/spectral.py`) | Reimplemented from Tran, Li & Madry, NeurIPS 2018 (no code copied) | Applied to last-prompt-token activations of an instruction-tuned LM |
| LoRA SVD weight-space probe (`src/detection/lora_svd_probe.py`) | Proposed in this project | New: energy fraction of ΔW = (α/r)BA along the known trigger's input direction, z-scored against random tokens |
| ONION perplexity filter | github.com/thunlp/ONION (not yet used) | Planned; commit to be pinned if adopted |
| Neural Cleanse | github.com/bolunwang/backdoor (not used) | Declared only; finite-label-set assumption discussed in report |

## AI tool declaration

Claude Code (Claude Opus 5.5) was used on Nov 5 2026 to scaffold this repository: the data,
poisoning, training, evaluation and detection modules, the unit tests, Slurm scripts, README
and this file. The experimental design follows the approved proposal. All reported numbers
must be produced by running this code; the author is responsible for reading, understanding
and being able to explain every file before the presentation.

## Change log of AI-assisted work

| Date | Commit | What was AI-assisted |
|---|---|---|
| 2026-11-05 | (initial) | Repository scaffold, Phase 1–2 pipeline, tests, smoke test |
| 2026-11-12/13 | (GAIVI fixes) | Conda-based Python 3.12 on GAIVI, tracked `logs/` dir for Slurm output |
| 2026-11-13 | (Colab runner) | `notebooks/colab_runner.ipynb`; allow `results_root` outside the repo |
| 2026-11-13 | (Kaggle runner) | `notebooks/kaggle_runner.ipynb` (platform switch from GAIVI to Kaggle T4) |
