"""Shared helpers: config loading, seeding, run directories, provenance logging.

Every run writes its fully-resolved config, environment, and git commit next to its
metrics so any number on a slide can be traced back to the exact run that produced it.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- config
def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge `upd` into `base` (in place) and return `base`."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _load_yaml_with_inherit(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = _load_yaml_with_inherit((path.parent / parent).resolve())
        cfg = deep_update(base, cfg)
    return cfg


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set cfg['a']['b'] = value for key 'a.b'. Unknown keys raise, so typos fail loudly."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown config section '{k}' in override '{dotted_key}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted_key}'")
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    """Load a YAML config (following `inherit:`) and apply `key.sub=value` overrides."""
    cfg = _load_yaml_with_inherit(Path(path).resolve())
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must look like key=value, got '{ov}'")
        key, raw = ov.split("=", 1)
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


# ----------------------------------------------------------------------------- seeding
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- run dirs
def make_run_dir(root: str | Path, name: str) -> Path:
    """Create a fresh, never-reused directory `<root>/<timestamp>_<name>[_k]`."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = root / f"{stamp}_{name}"
    d, k = base, 1
    while d.exists():
        d = Path(f"{base}_{k}")
        k += 1
    d.mkdir(parents=True, exist_ok=False)
    return d


def git_info() -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:  # not a git repo / git missing
        return {"commit": "unknown", "dirty": None}


def env_info() -> dict:
    import peft
    import transformers

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    if torch.cuda.is_available():
        info.update(
            gpu_name=torch.cuda.get_device_name(0),
            gpu_capability=list(torch.cuda.get_device_capability(0)),
            cuda_version=torch.version.cuda,
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    return info


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)
