"""Run ONE experiment: build data -> train -> evaluate -> detect -> save everything.

    python scripts/run_experiment.py --config configs/base.yaml --set attack.n_poison=20 --set seed=1

Outputs go to a NEW directory under `results_root` (never overwritten):
    config.yaml        fully-resolved config
    env.json           versions, GPU, git commit, Slurm job id
    train_log.jsonl    loss / grad-norm / lr every `log_every` steps
    generations.jsonl  every probe and clean generation (so metrics can be re-derived)
    metrics.json       all scalar metrics
    spectral.json      spectral-signature detection result (if enabled and n_poison > 0)
    adapter/           LoRA adapter weights (if LoRA and save_adapter)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.dolly import get_or_create_splits, load_dolly  # noqa: E402
from src.data.format import load_tokenizer, trigger_token_ids  # noqa: E402
from src.data.poison import add_trigger, build_train_set  # noqa: E402
from src.detection.spectral import evaluate_detector, prompt_representations, spectral_scores  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.training.train import build_model, count_params, train  # noqa: E402
from src.utils import env_info, get_device, git_info, load_config, make_run_dir, save_json, set_seed  # noqa: E402


def auto_name(cfg: dict) -> str:
    arm = cfg["arm"]
    a = f"lora-r{arm['rank']}" if arm["type"] == "lora" else f"full-lr{cfg['train']['lr']:g}"
    return f"{a}_n{cfg['attack']['n_poison']}_s{cfg['seed']}_{cfg['tag']}"


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. attack.n_poison=20")
    ap.add_argument("--skip-existing", action="store_true",
                    help="exit without training if a FINISHED run with the same name already exists "
                         "(lets a requeued Slurm job resume a sweep; unfinished runs are re-done in a new dir)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    name = cfg["run_name"] or auto_name(cfg)
    if args.skip_existing:
        done = sorted((ROOT / cfg["results_root"]).glob(f"*_{name}/metrics.json"))
        if done:
            print(f"[skip] {name} already finished: {done[-1].parent}")
            return done[-1].parent
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    use_amp = device.type == "cuda" and bool(cfg["train"]["fp16"])
    run_dir = make_run_dir(ROOT / cfg["results_root"], name)
    print(f"[run] {run_dir}")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    save_json({**env_info(), "git": git_info(), "device": str(device), "argv": sys.argv}, run_dir / "env.json")
    t_start = time.time()

    # ---------------------------------------------------------------- data
    records = load_dolly()
    by_id = {r["id"]: r for r in records}
    manifest = get_or_create_splits(records, cfg["data"], cfg["attack"]["reserved_strings"], ROOT / cfg["split_dir"])
    S = manifest["splits"]
    n_poison = int(cfg["attack"]["n_poison"])
    train_set = build_train_set(by_id, S, n_poison, int(cfg["data"]["pool_size"]), cfg["attack"])
    clean_test = [by_id[i] for i in S["clean_test"][: int(cfg["eval"]["clean_n"])]]
    probe = [add_trigger(by_id[i], cfg["attack"]) for i in S["trigger_probe"][: int(cfg["eval"]["probe_n"])]]
    val = [by_id[i] for i in S["val"]]
    print(f"[data] train={len(train_set)} (poison={n_poison}) clean_test={len(clean_test)} probe={len(probe)} "
          f"splits={Path(manifest['path']).name}")

    # ---------------------------------------------------------------- model + train
    tok = load_tokenizer(cfg["model"]["name"], cfg["model"].get("revision"))
    trig_ids = sorted({t for ex in probe for t in trigger_token_ids(tok, ex, cfg["attack"]["trigger"])})
    model = build_model(cfg)
    params = count_params(model)
    print(f"[model] {cfg['model']['name']} arm={cfg['arm']['type']} trainable={params['trainable']:,} "
          f"({100 * params['trainable_frac']:.3f}%) device={device} fp16={use_amp}")
    train_stats = train(model, tok, train_set, cfg, device, int(cfg["seed"]), run_dir / "train_log.jsonl")

    # ---------------------------------------------------------------- evaluation
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_eval = time.time()
    E = cfg["eval"]
    max_len = int(cfg["train"]["max_seq_len"])
    clean = M.response_loss(model, tok, clean_test, max_len, E["batch_size"], device, use_amp)
    val_loss = M.response_loss(model, tok, val, max_len, E["batch_size"], device, use_amp) if E["val_loss"] else None
    probe_gen = M.generate(model, tok, probe, int(E["probe_max_new_tokens"]), E["batch_size"], device, use_amp)
    clean_gen = M.generate(model, tok, clean_test, int(E["rouge_max_new_tokens"]), E["batch_size"], device, use_amp)
    asr = M.attack_success(probe_gen, cfg["attack"]["sentinel"])
    eval_time = time.time() - t_eval
    with open(run_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for ex, g in zip(probe, probe_gen):
            f.write(json.dumps({"split": "trigger_probe", "id": ex["id"], "instruction": ex["instruction"],
                                "context": ex["context"], "generation": g}) + "\n")
        for ex, g in zip(clean_test, clean_gen):
            f.write(json.dumps({"split": "clean_test", "id": ex["id"], "generation": g,
                                "reference": ex["response"]}) + "\n")

    metrics = {
        # relative to the repo when inside it; absolute when results_root is elsewhere (e.g. Google Drive)
        "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
        "model": cfg["model"]["name"], "arm": cfg["arm"]["type"],
        "rank": cfg["arm"]["rank"] if cfg["arm"]["type"] == "lora" else None,
        "lr": cfg["train"]["lr"], "seed": cfg["seed"], "n_poison": n_poison,
        "poison_rate": n_poison / len(train_set), "trigger": cfg["attack"]["trigger"],
        "trigger_token_ids": trig_ids, "placement": cfg["attack"]["placement"], "tag": cfg["tag"],
        **asr,
        "false_trigger_rate": M.false_trigger_rate(clean_gen, cfg["attack"]["sentinel"]),
        "clean_loss": clean["loss"], "clean_ppl": clean["ppl"],
        "val_loss": val_loss["loss"] if val_loss else None,
        "rougeL": M.rouge_l(clean_gen, [ex["response"] for ex in clean_test]),
        "params": params, **train_stats,
        "eval_time_s": round(eval_time, 2),
    }
    if device.type == "cuda":
        metrics["peak_gpu_mem_gb_eval"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)

    # ---------------------------------------------------------------- detection
    if cfg["detection"]["spectral"] and n_poison > 0:
        t_det = time.time()
        R = prompt_representations(model, tok, train_set, E["batch_size"], device, use_amp,
                                   layer=int(cfg["detection"]["spectral_layer"]))
        is_p = np.array([ex["is_poison"] for ex in train_set])
        sc = spectral_scores(R)
        det = evaluate_detector(sc, is_p, n_flag=int(round(1.5 * n_poison)))
        det["time_s"] = round(time.time() - t_det, 2)
        save_json(det, run_dir / "spectral.json")
        metrics.update({f"spectral_{k}": v for k, v in det.items()})

    if cfg["arm"]["type"] == "lora" and cfg["save_adapter"]:
        model.save_pretrained(run_dir / "adapter")
    if cfg["arm"]["type"] == "full" and cfg["save_full_model"]:
        model.save_pretrained(run_dir / "model")
    metrics["total_time_s"] = round(time.time() - t_start, 2)
    save_json(metrics, run_dir / "metrics.json")
    print(f"[done] ASR={metrics['asr']:.3f} (exact {metrics['asr_exact']:.3f}) "
          f"false-trigger={metrics['false_trigger_rate']:.3f} clean_loss={metrics['clean_loss']:.4f} "
          f"ROUGE-L={metrics['rougeL']:.4f} train={train_stats['train_time_s']}s eval={metrics['eval_time_s']}s")
    if "spectral_tpr" in metrics:
        print(f"[spectral] TPR={metrics['spectral_tpr']:.3f} FPR={metrics['spectral_fpr']:.4f} "
              f"AUROC={metrics['spectral_auroc']:.3f}")
    return run_dir


if __name__ == "__main__":
    main()
