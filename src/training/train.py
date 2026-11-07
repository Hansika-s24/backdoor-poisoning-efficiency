"""Model construction (LoRA or full fine-tuning) and a plain supervised fine-tuning loop.

Deliberately a hand-written loop rather than HF Trainer, so every step can be explained:

    for each optimizer step:
        take `micro_batch_size * grad_accum` examples (order = randperm(seed))
        loss = sum of token CE over response tokens / number of response tokens in the step
        backward (fp16 autocast + GradScaler on CUDA, fp32 on CPU)
        clip grad-norm, AdamW step, linear warmup/decay

Both arms load the base model in fp32 and run the SAME loop; the only difference is which
parameters have requires_grad=True (LoRA A/B matrices vs. every weight).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.format import IGNORE, collate, tokenize_example


def build_model(cfg: dict):
    from transformers import AutoModelForCausalLM

    mcfg, arm = cfg["model"], cfg["arm"]
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"], revision=mcfg.get("revision"), torch_dtype=torch.float32,
        attn_implementation=mcfg.get("attn_implementation", "sdpa"),
    )
    if arm["type"] == "lora":
        from peft import LoraConfig, get_peft_model

        r = int(arm["rank"])
        lcfg = LoraConfig(
            r=r,
            lora_alpha=float(arm["alpha_multiplier"]) * r,
            lora_dropout=float(arm["dropout"]),
            target_modules=list(arm["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
    elif arm["type"] == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"Unknown arm type {arm['type']}")
    return model


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_frac": trainable / total}


def response_ce_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum of next-token cross-entropy over positions whose label != -100.

    logits[:, t] predicts token t+1, hence the shift.
    """
    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=IGNORE, reduction="sum")
    return loss, int((targets != IGNORE).sum())


def train(model, tok, examples: list[dict], cfg: dict, device: torch.device, seed: int,
          log_path: Path) -> dict:
    tcfg = cfg["train"]
    feats = [tokenize_example(tok, ex, int(tcfg["max_seq_len"])) for ex in examples]
    n_trunc = sum(f["truncated"] for f in feats)

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(feats), generator=g).tolist()
    micro, accum = int(tcfg["micro_batch_size"]), int(tcfg["grad_accum"])
    eff = micro * accum
    steps = math.ceil(len(feats) * float(tcfg["epochs"]) / eff)
    # Multi-epoch: concatenate fresh permutations.
    while len(order) < steps * eff:
        order += torch.randperm(len(feats), generator=g).tolist()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]),
                            betas=(0.9, 0.999), eps=1e-8)
    from transformers import get_linear_schedule_with_warmup

    sched = get_linear_schedule_with_warmup(opt, int(round(float(tcfg["warmup_ratio"]) * steps)), steps)
    use_amp = device.type == "cuda" and bool(tcfg["fp16"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.to(device)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    log_every = int(tcfg.get("log_every", 10))
    running, skipped_steps = [], 0
    with open(log_path, "w", encoding="utf-8") as logf:
        for step in range(steps):
            idx = order[step * eff:(step + 1) * eff]
            batch_feats = [feats[i] for i in idx]
            n_tok = sum(sum(1 for y in f["labels"][1:] if y != IGNORE) for f in batch_feats)
            step_loss = 0.0
            for m in range(0, len(batch_feats), micro):
                b = collate(batch_feats[m:m + micro], tok.pad_token_id)
                b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                loss_sum, _ = response_ce_sum(logits, b["labels"])
                loss = loss_sum / max(n_tok, 1)  # token-weighted mean over the whole step
                scaler.scale(loss).backward()
                step_loss += loss.item()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, float(tcfg["max_grad_norm"]))
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                skipped_steps += 1  # inf/nan grads -> GradScaler skipped this update
            sched.step()
            opt.zero_grad(set_to_none=True)
            running.append(step_loss)
            if not math.isfinite(step_loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            if step % log_every == 0 or step == steps - 1:
                rec = {"step": step, "loss": step_loss, "grad_norm": float(gnorm), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0, 2)}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                print(f"  step {step + 1}/{steps} loss {step_loss:.4f} gnorm {float(gnorm):.3f}", flush=True)
    train_time = time.time() - t0
    stats = {
        "train_examples": len(feats),
        "train_poison_examples": sum(f["is_poison"] for f in feats),
        "train_truncated_examples": n_trunc,
        "optimizer_steps": steps,
        "effective_batch_size": eff,
        "train_time_s": round(train_time, 2),
        "sec_per_step": round(train_time / steps, 4),
        "final_train_loss_last10": sum(running[-10:]) / len(running[-10:]),
        "amp_fp16": use_amp,
        "amp_skipped_steps": skipped_steps,
    }
    if device.type == "cuda":
        stats["peak_gpu_mem_gb_train"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
    return stats
