"""Pre-download the dataset and model weights into the Hugging Face cache.

Run this on a node WITH internet (e.g. the GAIVI login node) before submitting GPU jobs,
in case compute nodes cannot reach huggingface.co.

    python scripts/download_assets.py --models Qwen/Qwen2.5-0.5B HuggingFaceTB/SmolLM2-135M
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dolly import dataset_hash, load_dolly  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-135M"])
    args = ap.parse_args()
    recs = load_dolly()
    print(f"dolly-15k: {len(recs)} records, content hash {dataset_hash(recs)}")
    from huggingface_hub import snapshot_download

    for m in args.models:
        path = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
        print(f"{m}: {path}")


if __name__ == "__main__":
    main()
