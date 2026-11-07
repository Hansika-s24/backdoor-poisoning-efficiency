"""Collect every finished run into reproducible summary CSVs.

    python scripts/summarize.py --runs results/runs --out results

Writes:
    runs.csv      one row per run (all scalar metrics + identifying config)
    grouped.csv   mean / std / count over seeds for each (setting, n_poison)
    n90.csv       per-seed n90 (first and stable crossing) and mean/std across seeds;
                  seeds that never reach ASR >= 0.9 are counted, not dropped or extrapolated
Nothing is overwritten except these three derived files, which are pure functions of the runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.metrics import n_threshold  # noqa: E402

SETTING = ["model", "arm", "rank", "lr", "trigger", "placement", "tag"]
METRICS = ["asr", "asr_exact", "false_trigger_rate", "clean_loss", "val_loss", "rougeL", "train_time_s",
           "eval_time_s", "peak_gpu_mem_gb_train", "spectral_tpr", "spectral_fpr", "spectral_auroc"]


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        rows.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rank"] = df["rank"].fillna(-1).astype(int)  # -1 = full fine-tuning
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    df = load_runs(ROOT / args.runs)
    if df.empty:
        sys.exit(f"No runs found under {args.runs}")
    df.to_csv(out / "runs.csv", index=False)

    dup = df.duplicated(SETTING + ["n_poison", "seed"], keep=False)
    if dup.any():
        print(f"WARNING: {int(dup.sum())} runs share setting+n+seed (re-runs?). They are all kept in runs.csv; "
              "grouped stats average over them. Inspect before reporting.")

    present = [c for c in METRICS if c in df.columns]
    g = df.groupby(SETTING + ["n_poison"], dropna=False)[present].agg(["mean", "std", "count"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g.reset_index().to_csv(out / "grouped.csv", index=False)

    n90_rows = []
    for key, sub in df.groupby(SETTING, dropna=False):
        per_seed = []
        for seed, s in sub.groupby("seed"):
            thr = n_threshold(s.groupby("n_poison")["asr"].mean().to_dict(), args.threshold)
            per_seed.append({"seed": seed, **thr})
        firsts = [r["first"] for r in per_seed if r["first"] is not None]
        stables = [r["stable"] for r in per_seed if r["stable"] is not None]
        n90_rows.append({
            **dict(zip(SETTING, key)),
            "n_seeds": len(per_seed),
            "seeds_reaching": len(firsts),
            "n90_first_mean": np.mean(firsts) if firsts else np.nan,
            "n90_first_std": np.std(firsts, ddof=1) if len(firsts) > 1 else np.nan,
            "n90_stable_mean": np.mean(stables) if stables else np.nan,
            "per_seed": json.dumps(per_seed, default=int),
        })
    pd.DataFrame(n90_rows).to_csv(out / "n90.csv", index=False)
    print(f"{len(df)} runs -> {out / 'runs.csv'}, grouped.csv, n90.csv")
    print(pd.DataFrame(n90_rows)[SETTING[1:4] + ["n_seeds", "seeds_reaching", "n90_first_mean", "n90_first_std"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
