"""EXPLORATORY follow-up to the SVD probe (not the pre-registered score).

For each seed, compares every per-module trigger z-score of a poisoned adapter with the SAME
module of that seed's clean (n = 0) adapter. Same seed = same LoRA initialisation and the same
training order apart from the replaced examples, so the difference isolates what the poisons
changed. Reads the svd_probe.json files written by scripts/run_svd_probe.py.

    python scripts/svd_probe_paired.py --runs results/kaggle/seed2/backdoor_results/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(runs: Path) -> dict:
    out = {}
    for f in runs.glob("*/svd_probe.json"):
        m = json.loads((f.parent / "metrics.json").read_text())
        rows = json.loads(f.read_text())["rows"]
        out[(m["seed"], m["n_poison"])] = (m["asr"], {(r["layer"], r["proj"], r["token_id"]): r["z"] for r in rows})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/kaggle/seed2/backdoor_results/runs")
    a = ap.parse_args()
    data = load(Path(a.runs))
    print("seed  n   ASR    max dz  where           mean dz  sd dz  #|dz|>2")
    for seed in sorted({s for s, _ in data}):
        _, z0 = data[(seed, 0)]
        keys = sorted(z0)
        for n in sorted(k for s, k in data if s == seed and k > 0):
            asr, zn = data[(seed, n)]
            dz = np.array([zn[k] - z0[k] for k in keys])
            i = int(np.argmax(dz))
            L, p, t = keys[i]
            print(f"{seed:>4} {n:>3} {asr:5.3f}  {dz.max():6.2f}  L{L:<2} {p} tok {t:<6}  {dz.mean():7.3f}  {dz.std():5.3f}  "
                  f"{int((np.abs(dz) > 2).sum())}")
    print(f"({len(keys)} module x trigger-token scores per adapter)")


if __name__ == "__main__":
    main()
