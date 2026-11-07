"""Verify the software/hardware stack before spending GPU time.

    python scripts/check_env.py [--model Qwen/Qwen2.5-0.5B]

Prints versions and GPU properties, reports bf16 support (expected: False on a Turing TITAN
RTX), and runs one fp16-autocast forward + backward pass of the model to check that fp16
gives finite loss and gradients -- Qwen2.5 was trained in bf16, so fp16 overflow is a real
risk that must be checked, not assumed away.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.utils import env_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    args = ap.parse_args()
    info = env_info()
    print(json.dumps(info, indent=2))
    if not torch.cuda.is_available():
        print("No CUDA device: only the CPU smoke test can run here.")
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).cuda()
    text = ["### Instruction:\nExplain photosynthesis briefly.\n\n### Response:\nPlants convert light into "
            "chemical energy, producing sugar and oxygen."] * 8
    b = tok(text, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(**b, labels=b["input_ids"])
    out.loss.backward()
    torch.cuda.synchronize()
    finite_grads = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"fp16 autocast forward+backward: loss={out.loss.item():.4f} finite_loss={torch.isfinite(out.loss).item()} "
          f"finite_grads={finite_grads} time={time.time() - t:.2f}s "
          f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    if not (torch.isfinite(out.loss).item() and finite_grads):
        print("WARNING: fp16 produced non-finite values; set train.fp16=false (fp32) and re-estimate cost.")


if __name__ == "__main__":
    main()
