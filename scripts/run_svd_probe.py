"""Apply the LoRA SVD weight-space probe to every saved adapter under a results directory.

    python scripts/run_svd_probe.py --runs results/runs --out results/svd_probe.csv

Runs on CPU (only the base embeddings + RMSNorm weights and the small adapters are needed).
Poisoned and clean (n_poison = 0) adapters are scored identically; the clean ones give the
null distribution of the model-level score S.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.format import load_tokenizer  # noqa: E402
from src.detection.lora_svd_probe import load_base_vectors, probe_adapter, sample_null_tokens  # noqa: E402
from src.utils import save_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/svd_probe.csv")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--null-seed", type=int, default=0)
    args = ap.parse_args(argv)

    run_dirs = sorted(p.parent.parent for p in (ROOT / args.runs).glob("*/adapter/adapter_config.json"))
    run_dirs = [d for d in run_dirs if (d / "metrics.json").exists()]
    if not run_dirs:
        sys.exit(f"No finished runs with adapters under {args.runs}")
    by_model: dict[str, list[Path]] = {}
    for d in run_dirs:
        by_model.setdefault(json.loads((d / "metrics.json").read_text())["model"], []).append(d)

    rows = []
    for model_name, dirs in by_model.items():
        tok = load_tokenizer(model_name)
        embed, norms = load_base_vectors(model_name)
        for d in dirs:
            m = json.loads((d / "metrics.json").read_text())
            trig = m["trigger_token_ids"]
            null = sample_null_tokens(tok, args.n_null, set(trig), args.null_seed, embed.shape[0])
            res = probe_adapter(d / "adapter", embed, norms, trig, null)
            save_json(res, d / "svd_probe.json")
            rows.append({"run_dir": m["run_dir"], "model": model_name, "rank": res["rank"], "seed": m["seed"],
                         "n_poison": m["n_poison"], "asr": m["asr"], "trigger": m["trigger"],
                         "placement": m["placement"], "S_max_z": res["S_max_z"],
                         "mean_z_layer0": res["mean_z_layer0"],
                         **{f"argmax_{k}": v for k, v in res["argmax"].items()}})
            print(f"{m['run_dir']}: n={m['n_poison']} S_max_z={res['S_max_z']:.2f} argmax={res['argmax']}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
