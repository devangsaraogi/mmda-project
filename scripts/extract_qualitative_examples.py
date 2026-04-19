"""Extract concrete retrieved-image examples for the qualitative figure.

For 3 representative claims, runs every retriever we report in the paper
(global CLIP, per-claim CLIP, hybrid CLIP+caption-BM25) and saves:

  <out-dir>/qualitative.json           — claim text, gold IDs, top-5 from each retriever
  <out-dir>/images/<image_id>.jpg      — the actual images, extracted from the TSV

Run on the cluster (needs the TSV). Takes ~2–3 minutes wall on the login
node — no GPU needed, just a per-claim CLIP and BM25 pass over small pools.

After it writes, scp the whole <out-dir>/ to local and run
``python scripts/build_qualitative_panel.py`` to assemble the HTML + PNG.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs
from src.models.clip_encoder import CLIPEncoder
from src.data.webqa_dataset import WebQADataset
from src.retrieval import EmbeddingIndex, CLIPRetriever
from src.retrieval_hybrid import CaptionBM25Retriever, RRFHybridRetriever


# Three claims I'd pick by default — mix of manipulation types. User can override
# with --claim-ids.
DEFAULT_CLAIM_IDS = [
    "webqaadv_000015",   # visual_contradiction candidate
    "webqaadv_000022",   # true with gold in pool
    "webqaadv_000030",   # unverifiable / empty pool failure case
]


def _pick_claims(dataset, ids_filter: list[str] | None, max_claims: int = 3):
    """Return up to `max_claims` claim items from the test split.

    If `ids_filter` is provided, pick those specific IDs.
    Otherwise, pick one per distinct manipulation_type with gold-in-pool where possible.
    """
    test = dataset.get_split("test")
    id_to_idx = {}
    for i in range(len(test)):
        id_to_idx[test[i]["claim_id"]] = i

    picked = []
    if ids_filter:
        for cid in ids_filter:
            if cid in id_to_idx:
                picked.append(test[id_to_idx[cid]])
        return picked[:max_claims]

    # Auto-pick: first True, first VisualContradiction, first empty-pool from test split.
    want = {"true": None, "visual_contradiction": None, "ambiguous": None}
    for i in range(len(test)):
        item = test[i]
        mt = item.get("misinfo_type", "")
        if mt in want and want[mt] is None:
            pool = item.get("image_candidate_ids", [])
            gold = item.get("gold_image_ids", [])
            if mt == "ambiguous" or (mt == "true" and gold and set(gold) & set(pool)) \
                    or (mt == "visual_contradiction" and gold and set(gold) & set(pool)):
                want[mt] = item
        if all(v is not None for v in want.values()):
            break
    picked = [v for v in want.values() if v is not None]
    return picked[:max_claims]


def _top_k_global(retriever: CLIPRetriever, claim_text: str, k: int):
    return retriever.retrieve(claim_text, top_k=k)


def _top_k_per_claim(retriever: CLIPRetriever, claim_text: str,
                     candidate_ids: list[str], k: int):
    return retriever.retrieve_scoped(claim_text, candidate_ids, top_k=k)


def _top_k_hybrid(hybrid: RRFHybridRetriever, claim_text: str,
                  candidate_ids: list[str], metadata: list[dict], k: int):
    return hybrid.retrieve_scoped(claim_text, candidate_ids, metadata, top_k=k)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--embeddings", type=str, required=True,
                    help="Path to pre-computed CLIP image_index.pt")
    ap.add_argument("--claim-ids", nargs="*", default=None,
                    help="Override: pick these specific claim IDs (default: auto-pick 3)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("results/qualitative"))
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)

    # Dataset + retrievers (CPU ok; the pools are tiny).
    dataset = WebQADataset(cfg)
    encoder = CLIPEncoder(cfg)
    index = EmbeddingIndex()
    index.load(args.embeddings)
    clip_retriever = CLIPRetriever(encoder, index, cfg)
    caption_retriever = CaptionBM25Retriever(default_top_k=args.k)
    hybrid = RRFHybridRetriever(clip_retriever, caption_retriever, k_rrf=60)

    claims = _pick_claims(dataset, args.claim_ids)
    if not claims:
        print("No claims picked. Either the IDs don't match or auto-picker failed.")
        return 1

    print(f"Picked {len(claims)} claims")
    out_payload = []

    for claim in claims:
        cid = claim["claim_id"]
        claim_text = claim["claim_text"]
        gold_ids = [str(g) for g in claim["gold_image_ids"]]
        cand_ids = [str(c) for c in claim.get("image_candidate_ids", []) if c is not None]
        metadata = claim.get("image_candidate_metadata", [])
        mtype = claim.get("misinfo_type", "unknown")

        print(f"\n--- {cid}  ({mtype})  gold={gold_ids}  pool={len(cand_ids)} ---")
        print(f"    {claim_text[:140]}")

        global_top = _top_k_global(clip_retriever, claim_text, args.k)
        per_claim_top = _top_k_per_claim(clip_retriever, claim_text, cand_ids, args.k)
        hybrid_top = _top_k_hybrid(hybrid, claim_text, cand_ids, metadata, args.k)

        print(f"    global    top-5: {[iid for iid, _ in global_top]}")
        print(f"    per-claim top-5: {[iid for iid, _ in per_claim_top]}")
        print(f"    hybrid    top-5: {[iid for iid, _ in hybrid_top]}")

        # Extract the union of image IDs we need to render.
        needed_ids = set()
        for iid, _ in global_top + per_claim_top + hybrid_top:
            needed_ids.add(str(iid))
        for g in gold_ids:
            needed_ids.add(str(g))

        extracted = []
        for iid in needed_ids:
            img = dataset.get_image(iid)
            if img is None:
                continue
            out_path = args.out_dir / "images" / f"{iid}.jpg"
            try:
                img.convert("RGB").save(out_path, "JPEG", quality=85)
                extracted.append(iid)
            except Exception as e:
                print(f"    WARN: could not save image {iid}: {e}")

        # Metadata for the panel assembly, including title+caption for each image.
        meta_by_id = {}
        for m in metadata:
            mid = str(m.get("id", ""))
            if mid:
                meta_by_id[mid] = {
                    "title": m.get("title", ""),
                    "caption": m.get("caption", ""),
                }

        out_payload.append({
            "claim_id": cid,
            "claim_text": claim_text,
            "misinfo_type": mtype,
            "gold_image_ids": gold_ids,
            "candidate_pool_size": len(cand_ids),
            "retrieved": {
                "global":    [{"id": iid, "score": float(s)} for iid, s in global_top],
                "per_claim": [{"id": iid, "score": float(s)} for iid, s in per_claim_top],
                "hybrid":    [{"id": iid, "score": float(s)} for iid, s in hybrid_top],
            },
            "image_metadata": meta_by_id,
            "extracted_images": extracted,
        })

    payload_path = args.out_dir / "qualitative.json"
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote {payload_path}")
    print(f"Wrote {sum(len(c['extracted_images']) for c in out_payload)} images under {args.out_dir / 'images'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
