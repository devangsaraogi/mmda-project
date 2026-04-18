"""Enrich webqa_adv.jsonl with per-claim image candidate IDs.

Joins each adversarial claim (by source_webqa_id) to the original WebQA
annotation file and pulls the integer image_ids from img_posFacts and
img_negFacts. The result is a new JSONL with extra fields:

    image_candidate_pos_ids: [str, ...]   # integer IDs, from img_posFacts
    image_candidate_neg_ids: [str, ...]   # integer IDs, from img_negFacts
    image_candidate_ids:     [str, ...]   # pos + neg, preserving order

Run on the cluster where WebQA_train_val.json lives.

Usage:
    python scripts/enrich_webqa_adv_candidates.py \\
        --adv-in  /share/csc791003s26/cmvn/dataset-folder/step0_out/webqa_adv.jsonl \\
        --webqa   /share/csc791003s26/cmvn/dataset-folder/extracted_webqa_data/WebQA_train_val.json \\
        --out     /share/csc791003s26/cmvn/dataset-folder/step0_out/webqa_adv_with_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_webqa_originals(path: Path) -> dict:
    logger.info("Loading WebQA originals from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d WebQA records", len(data))
    return data


def ids_from_facts(facts) -> list[str]:
    out = []
    if not isinstance(facts, list):
        return out
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        iid = fact.get("image_id")
        if iid is None:
            continue
        out.append(str(iid))
    return out


def enrich(adv_in: Path, webqa_path: Path, out: Path) -> None:
    orig = load_webqa_originals(webqa_path)

    enriched = missing = no_candidates = 0
    total = 0
    pos_histogram: dict[int, int] = {}
    neg_histogram: dict[int, int] = {}

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(adv_in, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)

            src_id = obj.get("source_webqa_id")
            pos_ids: list[str] = []
            neg_ids: list[str] = []

            if src_id and src_id in orig:
                orig_rec = orig[src_id]
                pos_ids = ids_from_facts(orig_rec.get("img_posFacts"))
                neg_ids = ids_from_facts(orig_rec.get("img_negFacts"))
                enriched += 1
            elif src_id:
                missing += 1
            else:
                missing += 1

            if not pos_ids and not neg_ids:
                no_candidates += 1

            pos_histogram[len(pos_ids)] = pos_histogram.get(len(pos_ids), 0) + 1
            neg_histogram[len(neg_ids)] = neg_histogram.get(len(neg_ids), 0) + 1

            obj["image_candidate_pos_ids"] = pos_ids
            obj["image_candidate_neg_ids"] = neg_ids
            # Ordered: positives first, then negatives, dedup preserving order
            seen: set[str] = set()
            combined: list[str] = []
            for iid in pos_ids + neg_ids:
                if iid in seen:
                    continue
                seen.add(iid)
                combined.append(iid)
            obj["image_candidate_ids"] = combined

            fout.write(json.dumps(obj) + "\n")

    logger.info("=" * 50)
    logger.info("ENRICHMENT SUMMARY")
    logger.info(f"  Total claims:                   {total}")
    logger.info(f"  Enriched (source_webqa_id hit): {enriched}")
    logger.info(f"  Missing (no source link):       {missing}")
    logger.info(f"  Claims with no candidates:      {no_candidates}")
    logger.info(f"  Wrote: {out}")
    logger.info("Pos-id histogram (count -> claims): %s",
                 dict(sorted(pos_histogram.items())))
    logger.info("Neg-id histogram (count -> claims): %s",
                 dict(sorted(neg_histogram.items())))
    logger.info("=" * 50)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adv-in", type=Path, required=True)
    ap.add_argument("--webqa", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if not args.adv_in.exists():
        logger.error("adv-in does not exist: %s", args.adv_in)
        return 2
    if not args.webqa.exists():
        logger.error("webqa does not exist: %s", args.webqa)
        return 2

    enrich(args.adv_in, args.webqa, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
