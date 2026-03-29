"""Manual data exploration: stratified sampling of 30 examples for report.

Requirement A1: Examine 20-30 examples, annotate modality interactions,
identify biases, and motivate multimodal approach.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed
from src.data.webqa_dataset import WebQADataset


def stratified_sample(dataset, n_per_type=6, seed=42):
    """Sample n_per_type examples from each misinformation type.

    Returns list of dicts with claim metadata.
    """
    rng = np.random.RandomState(seed)

    # Group by misinfo type
    by_type = defaultdict(list)
    for i in range(len(dataset)):
        item = dataset[i]
        by_type[item["misinfo_type"]].append(item)

    sampled = []
    for mtype in sorted(by_type.keys()):
        items = by_type[mtype]
        n = min(n_per_type, len(items))
        indices = rng.choice(len(items), size=n, replace=False)
        for idx in indices:
            sampled.append(items[idx])

    return sampled


def format_example(item, dataset, idx):
    """Format a single example for display and annotation."""
    # Get text candidate snippets (first 3)
    text_snippets = []
    for cand in item.get("text_candidates", [])[:3]:
        text = dataset.get_candidate_text(cand)
        cand_id = dataset.get_candidate_id(cand)
        text_snippets.append({"id": cand_id, "text": text[:200]})

    return {
        "index": idx,
        "claim_id": item["claim_id"],
        "claim_text": item["claim_text"],
        "label": ["True", "False", "Unverifiable"][item["label"]],
        "misinfo_type": item["misinfo_type"],
        "gold_image_ids": item["gold_image_ids"][:5],
        "gold_text_ids": item["gold_text_ids"][:5],
        "n_gold_images": len(item["gold_image_ids"]),
        "n_gold_texts": len(item["gold_text_ids"]),
        "n_text_candidates": len(item.get("text_candidates", [])),
        "text_snippets_preview": text_snippets,
        # Annotation fields (to be filled manually)
        "annotation": {
            "active_modalities": "",  # "visual", "text", "both"
            "modality_interaction": "",  # how modalities interact
            "unimodal_sufficient": None,  # True/False
            "biases_noted": "",  # linguistic shortcuts, visual framing
            "external_knowledge_required": None,  # True/False
            "notes": "",
        },
    }


def print_example(ex):
    """Pretty-print one example to console."""
    print(f"\n{'='*70}")
    print(f"[{ex['index']}] Claim ID: {ex['claim_id']}")
    print(f"    Type: {ex['misinfo_type']} | Label: {ex['label']}")
    print(f"    Claim: {ex['claim_text']}")
    print(f"    Gold images: {ex['n_gold_images']} IDs: {ex['gold_image_ids']}")
    print(f"    Gold texts:  {ex['n_gold_texts']} IDs: {ex['gold_text_ids']}")
    print(f"    Text candidates: {ex['n_text_candidates']}")
    if ex["text_snippets_preview"]:
        print("    Text evidence preview:")
        for snippet in ex["text_snippets_preview"]:
            print(f"      [{snippet['id']}] {snippet['text'][:120]}...")
    print(f"{'='*70}")


def compute_summary_stats(examples):
    """Compute summary statistics across sampled examples."""
    stats = {
        "total": len(examples),
        "by_type": defaultdict(int),
        "by_label": defaultdict(int),
        "avg_gold_images": 0,
        "avg_gold_texts": 0,
        "avg_text_candidates": 0,
        "has_both_modalities": 0,
        "image_only": 0,
        "text_only": 0,
    }

    for ex in examples:
        stats["by_type"][ex["misinfo_type"]] += 1
        stats["by_label"][ex["label"]] += 1
        stats["avg_gold_images"] += ex["n_gold_images"]
        stats["avg_gold_texts"] += ex["n_gold_texts"]
        stats["avg_text_candidates"] += ex["n_text_candidates"]

        has_img = ex["n_gold_images"] > 0
        has_txt = ex["n_gold_texts"] > 0
        if has_img and has_txt:
            stats["has_both_modalities"] += 1
        elif has_img:
            stats["image_only"] += 1
        elif has_txt:
            stats["text_only"] += 1

    n = len(examples)
    stats["avg_gold_images"] = round(stats["avg_gold_images"] / n, 2) if n else 0
    stats["avg_gold_texts"] = round(stats["avg_gold_texts"] / n, 2) if n else 0
    stats["avg_text_candidates"] = round(stats["avg_text_candidates"] / n, 2) if n else 0
    stats["by_type"] = dict(stats["by_type"])
    stats["by_label"] = dict(stats["by_label"])

    return stats


def main():
    parser = argparse.ArgumentParser(description="Manual data exploration for report")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n-per-type", type=int, default=6,
                        help="Number of examples per misinformation type")
    parser.add_argument("--output", type=str, default="results/manual_exploration.json",
                        help="Output JSON path")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to sample from")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.seed)

    print("Loading dataset...")
    dataset = WebQADataset(cfg)
    split_data = dataset.get_split(args.split)
    print(f"Split '{args.split}': {len(split_data)} claims")

    print(f"\nSampling {args.n_per_type} examples per misinformation type...")
    sampled = stratified_sample(split_data, n_per_type=args.n_per_type, seed=cfg.seed)

    # Format examples
    examples = []
    for i, item in enumerate(sampled):
        ex = format_example(item, dataset, i + 1)
        examples.append(ex)
        print_example(ex)

    # Summary stats
    stats = compute_summary_stats(examples)
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total examples: {stats['total']}")
    print(f"By type: {json.dumps(stats['by_type'], indent=2)}")
    print(f"By label: {json.dumps(stats['by_label'], indent=2)}")
    print(f"Avg gold images: {stats['avg_gold_images']}")
    print(f"Avg gold texts: {stats['avg_gold_texts']}")
    print(f"Avg text candidates: {stats['avg_text_candidates']}")
    print(f"Both modalities: {stats['has_both_modalities']}")
    print(f"Image only: {stats['image_only']}")
    print(f"Text only: {stats['text_only']}")

    # Save
    output = {
        "metadata": {
            "split": args.split,
            "n_per_type": args.n_per_type,
            "seed": cfg.seed,
            "total_in_split": len(split_data),
        },
        "summary_stats": stats,
        "examples": examples,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
