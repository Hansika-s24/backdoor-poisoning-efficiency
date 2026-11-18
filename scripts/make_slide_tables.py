"""Table images for the slides (navy header to match the slide template).

Numbers are copied from docs/rank_ablation_2026-11-17.md, docs/svd_probe_2026-11-17.md and
docs/sweep_full_ft_2026-11-17.md (all produced by scripts/summarize.py / run_svd_probe.py).

    python scripts/make_slide_tables.py  ->  results/figures/slides/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "slides"
NAVY, INK, ROW, ROW2 = "#12306b", "#0b0b0b", "#ffffff", "#e8eef8"


def table(name, header, rows, col_w, bold_cells=(), fs=15, row_h=0.62):
    w, h = sum(col_w), row_h * (len(rows) + 1)
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    for r, cells in enumerate([header] + rows):
        y = h - (r + 1) * row_h
        x = 0
        for c, (txt, cw) in enumerate(zip(cells, col_w)):
            fc = NAVY if r == 0 else (ROW if r % 2 else ROW2)
            ax.add_patch(plt.Rectangle((x, y), cw, row_h, fc=fc, ec="white", lw=2))
            bold = r == 0 or (r, c) in bold_cells or c == 0
            ax.text(x + cw / 2, y + row_h / 2, txt, ha="center", va="center", fontsize=fs,
                    color="white" if r == 0 else INK, fontweight="bold" if bold else "normal")
            x += cw
    fig.savefig(OUT / f"{name}.png", transparent=True)
    plt.close(fig)
    print("wrote", name)


OUT.mkdir(parents=True, exist_ok=True)
table("setup_arms", ["", "LoRA r = 8", "Full fine-tuning"],
      [["Trainable params", "1.08 M (q/k/v/o)", "494 M (all)"],
       ["Learning rate", "2e-4", "1e-5 (matched)"],
       ["Val loss at n = 0", "2.104", "2.098"],
       ["Steps / seeds", "313 / 3", "313 / 3"]],
      [2.6, 2.6, 2.6])
table("rank_ablation", ["LoRA rank r", "4", "8", "16", "32", "64"],
      [["ASR, n = 100", "0.000", "0.000", "0.000", "0.015", "0.975"],
       ["ASR, n = 200", "0.000", "0.975", "0.980", "0.995", "0.995"]],
      [2.2, 1.1, 1.1, 1.1, 1.1, 1.1], bold_cells={(1, 5), (2, 1)})
table("svd_probe", ["Probe score S", "clean (n = 0)", "backdoored (n = 200)"],
      [["Seed 1", "5.58", "5.01"], ["Seed 2", "4.83", "4.58"]],
      [2.2, 2.4, 2.9])
table("n90", ["n90 (poisons for ASR ≥ 0.9)", "seed 0", "seed 1", "seed 2"],
      [["LoRA r = 8", "200", "200", "200"], ["Full fine-tuning", "100", "100", "50"]],
      [3.6, 1.3, 1.3, 1.3])
