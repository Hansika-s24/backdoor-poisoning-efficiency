"""Slide figures for the main comparison: ASR vs n and spectral AUROC vs n, LoRA r=8 vs full FT.

Reads results/runs.csv (from scripts/summarize.py). One point per (arm, seed, n); the line is the
mean over the seeds that exist for that arm, so a 1-seed arm is visibly a single curve.

LoRA seed 0's run files were lost (docs/sweep_seed0_2026-11-15.md). Its values are transcribed
from the printed output below and are drawn with HOLLOW markers so they can't be mistaken for
values read from run files. Pass --no-transcribed to leave them out.

Usage:  python scripts/make_figures.py [--out results/figures] [--no-transcribed]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GRID = [0, 1, 2, 5, 10, 20, 50, 100, 200]

# Transcribed LoRA r=8 seed 0 values (docs/sweep_lora_r8_2026-11-15.md); NOT in runs.csv.
LORA_S0_ASR = {n: 0.0 for n in GRID} | {200: 0.955}
LORA_S0_AUROC = {1: 0.675, 2: 0.636, 5: 0.679, 10: 0.535, 20: 0.544, 50: 0.569, 100: 0.514, 200: 0.932}

# Reference palette (dataviz skill, light mode), slots 1-2 in fixed order.
ARMS = {
    "lora": dict(label="LoRA r=8 (lr 2e-4)", color="#2a78d6"),
    "full": dict(label="Full fine-tuning (lr 1e-5)", color="#eb6834"),
}
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"


def load(include_transcribed: bool) -> pd.DataFrame:
    df = pd.read_csv(REPO / "results" / "runs.csv")
    df = df[(df.tag == "main")
            & (((df.arm == "lora") & (df["rank"] == 8)) | ((df.arm == "full") & np.isclose(df.lr, 1e-5)))]
    df = df[["arm", "seed", "n_poison", "asr", "spectral_auroc"]].assign(transcribed=False)
    if include_transcribed:
        rows = [dict(arm="lora", seed=0, n_poison=n, asr=LORA_S0_ASR[n],
                     spectral_auroc=LORA_S0_AUROC.get(n, np.nan), transcribed=True) for n in GRID]
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    dup = df.duplicated(["arm", "seed", "n_poison"], keep=False)
    if dup.any():
        raise SystemExit(f"duplicate (arm, seed, n) rows:\n{df[dup]}")
    return df


def style(ax, ylabel: str, grid=GRID):
    ax.set_xticks(range(len(grid)), [str(n) for n in grid])
    ax.set_xlabel("poisoned examples n (out of 5,000)", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def plot(df: pd.DataFrame, col: str, ylabel: str, title: str, path: Path, ref: float | None, grid=GRID):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(-0.3, ref - 0.015, "ASR 0.9 (n90 threshold)" if ref == 0.9 else "chance (0.5)",
                color=INK2, fontsize=8, ha="left", va="top")
    jitter = {"lora": -0.08, "full": 0.08}
    for arm, spec in ARMS.items():
        d = df[(df.arm == arm) & df[col].notna()]
        if d.empty:
            continue
        x = d.n_poison.map(grid.index) + jitter[arm]
        filled = ~d.transcribed
        ax.scatter(x[filled], d[col][filled], s=22, color=spec["color"], alpha=0.55, lw=0, zorder=3)
        ax.scatter(x[~filled], d[col][~filled], s=22, facecolor="white", edgecolor=spec["color"],
                   lw=1.2, alpha=0.8, zorder=3)
        m = d.groupby("n_poison")[col].mean()
        k = d.seed.nunique()
        ax.plot([grid.index(n) + jitter[arm] for n in m.index], m.values, color=spec["color"], lw=2, zorder=4,
                label=f"{spec['label']}, {k} seed{'s' if k > 1 else ''}")
    style(ax, ylabel, grid)
    ax.set_xlim(-0.4, len(grid) - 0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    note = "Line = mean over seeds; dots = individual seeds."
    if df.transcribed.any():
        note += " Hollow = LoRA seed 0, transcribed from printed log (run files lost)."
    fig.text(0.01, 0.01, note, fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.relative_to(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--no-transcribed", action="store_true")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    df = load(not a.no_transcribed)
    print(df.groupby("arm").seed.unique().to_string())
    plot(df, "asr", "attack success rate (ASR)",
         "Backdoor success vs. number of poisoned examples (Qwen2.5-0.5B, dolly-15k)",
         out / "asr_vs_n.png", ref=0.9)
    plot(df[df.n_poison > 0], "spectral_auroc", "spectral signatures AUROC",
         "Spectral-signature detection of the poisoned examples",
         out / "spectral_auroc_vs_n.png", ref=0.5, grid=GRID[1:])


if __name__ == "__main__":
    main()
