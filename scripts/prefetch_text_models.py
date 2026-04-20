"""Prefetch HuggingFace models for the text ablation on a login node.

Compute nodes in the Hazel cluster have no internet; we cache to
``$HF_HOME`` (scratch) on the login node so the bsub jobs can run with
``HF_HUB_OFFLINE=1``. Run once before submitting the text-ablation bsubs.

Usage (login node):

    module load python/3.9.6
    cd ~/mmda-project
    source venv/bin/activate
    export HF_HOME=/share/csc791003s26/$USER/.cache/huggingface
    python scripts/prefetch_text_models.py
"""
from __future__ import annotations

import sys


MODELS = [
    # Dense retriever
    ("BAAI/bge-small-en-v1.5", "AutoModel"),
    # Cross-encoder reranker
    ("cross-encoder/ms-marco-MiniLM-L-12-v2", "AutoModelForSequenceClassification"),
    # NLI large (for scale ablation)
    ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli", "AutoModelForSequenceClassification"),
]


def main() -> int:
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    model_cls_map = {
        "AutoModel": AutoModel,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
    }
    for name, cls_name in MODELS:
        print(f"\n=== {name} ({cls_name}) ===")
        AutoTokenizer.from_pretrained(name)
        print("  tokenizer: ok")
        model_cls_map[cls_name].from_pretrained(name)
        print("  model: ok")

    print("\nAll models cached. Safe to submit with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
