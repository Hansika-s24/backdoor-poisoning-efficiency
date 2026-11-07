"""One-command end-to-end smoke test (CPU or GPU, < 10 min on a laptop CPU).

    python scripts/smoke_test.py

Runs configs/smoke.yaml (SmolLM2-135M, 64 training examples incl. 16 poisoned, LoRA r=8),
then the SVD probe on the resulting adapter, then checks every expected output file exists
and every metric is a finite number. Outputs go to results/smoke/ (git-ignored).
The numbers it prints test the plumbing only; they are not experimental results.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_experiment  # noqa: E402

from src.data.format import load_tokenizer  # noqa: E402
from src.detection.lora_svd_probe import load_base_vectors, probe_adapter, sample_null_tokens  # noqa: E402
from src.utils import save_json  # noqa: E402

EXPECTED = ["config.yaml", "env.json", "train_log.jsonl", "generations.jsonl", "metrics.json", "spectral.json",
            "adapter/adapter_config.json", "adapter/adapter_model.safetensors"]


def main():
    t0 = time.time()
    run_dir = run_experiment.main(["--config", str(ROOT / "configs" / "smoke.yaml"), *sys.argv[1:]])
    missing = [f for f in EXPECTED if not (run_dir / f).exists()]
    assert not missing, f"missing outputs: {missing}"
    m = json.loads((run_dir / "metrics.json").read_text())
    for k in ("asr", "asr_exact", "false_trigger_rate", "clean_loss", "rougeL", "spectral_auroc"):
        assert isinstance(m[k], (int, float)) and math.isfinite(m[k]), f"{k} not finite: {m[k]}"

    tok = load_tokenizer(m["model"])
    embed, norms = load_base_vectors(m["model"])
    null = sample_null_tokens(tok, 500, set(m["trigger_token_ids"]), 0, embed.shape[0])
    probe = probe_adapter(run_dir / "adapter", embed, norms, m["trigger_token_ids"], null)
    save_json(probe, run_dir / "svd_probe.json")
    assert math.isfinite(probe["S_max_z"])
    print(f"[svd-probe] S_max_z={probe['S_max_z']:.2f} at {probe['argmax']}")
    print(f"SMOKE TEST PASSED in {time.time() - t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
