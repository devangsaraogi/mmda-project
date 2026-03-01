"""End-to-end pipeline: precompute -> retrieval -> train MLP -> verification."""

import argparse
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.models.clip_encoder import CLIPEncoder
from src.data.webqa_dataset import WebQADataset
from src.retrieval import EmbeddingIndex, CLIPRetriever, precompute_embeddings, evaluate_retrieval
from src.evaluation import save_metrics
from src.visualization import plot_recall_at_k

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run full CLIP visual pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to pre-computed image_index.pt (skip embedding step)")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("full_pipeline", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        # === Step 1: Load data ===
        print("\n" + "=" * 60)
        print("STEP 1: Loading dataset")
        print("=" * 60)

        dataset = WebQADataset(cfg)

        train_size = len(dataset.get_split("train"))
        val_size = len(dataset.get_split("val"))
        test_size = len(dataset.get_split("test"))
        pool_size = len(dataset.get_all_image_ids())

        print(f"  Total claims: {len(dataset)}")
        print(f"  Image pool: {pool_size}")
        print(f"  Train/Val/Test: {train_size}/{val_size}/{test_size}")

        tracker.log_dataset(
            name="WebQA-Adv",
            total=len(dataset),
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            image_pool_size=pool_size,
        )

        # === Step 2: Pre-compute embeddings ===
        print("\n" + "=" * 60)
        print("STEP 2: Pre-computing CLIP embeddings")
        print("=" * 60)

        encoder = CLIPEncoder(cfg)

        if args.embeddings and os.path.exists(args.embeddings):
            print(f"  Loading pre-computed embeddings from {args.embeddings}")
            index = EmbeddingIndex()
            index.load(args.embeddings)
            index_path = args.embeddings
        else:
            index_path = os.path.join(cfg.results.embeddings_dir, "image_index.pt")
            index = precompute_embeddings(encoder, dataset, index_path, batch_size=cfg.clip.batch_size)

        tracker.log_step("embeddings", num_images=pool_size, path=index_path)

        # === Step 3: Retrieval evaluation ===
        print("\n" + "=" * 60)
        print("STEP 3: Retrieval evaluation (Recall@K)")
        print("=" * 60)

        retriever = CLIPRetriever(encoder, index, cfg)
        test_data = dataset.get_split("test")
        recall_results = evaluate_retrieval(retriever, test_data, cfg.retrieval.top_k_values)

        save_metrics(
            {"recall_at_k": recall_results},
            os.path.join(cfg.results.metrics_dir, "retrieval_metrics.json"),
        )
        plot_recall_at_k(
            recall_results,
            os.path.join(cfg.results.figures_dir, "recall_at_k.png"),
        )

        for k, r in sorted(recall_results.items()):
            print(f"  Recall@{k}: {r:.4f}")

        tracker.log_step("retrieval", **{f"recall@{k}": v for k, v in recall_results.items()})

        # === Step 4: Train MLP ===
        print("\n" + "=" * 60)
        print("STEP 4: Training MLP classifier")
        print("=" * 60)

        from scripts.train_mlp import train_mlp
        train_mlp(cfg, tracker=tracker, embedding_index=index)

        tracker.log_step("train_mlp", status="complete")

        # === Step 5: Verification ===
        print("\n" + "=" * 60)
        print("STEP 5: Running verification (threshold + MLP)")
        print("=" * 60)

        from scripts.run_verification import run_threshold_verification, run_mlp_verification

        print("\n--- Threshold-based ---")
        thresh_oracle, thresh_e2e = run_threshold_verification(cfg, encoder, dataset, retriever,
                                                               embedding_index=index)

        print("\n--- MLP-based ---")
        mlp_oracle, mlp_e2e = run_mlp_verification(cfg, encoder, dataset, retriever,
                                                    embedding_index=index)

        # === Summary ===
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE — Summary")
        print("=" * 60)

        print(f"\nRetrieval:")
        for k, r in sorted(recall_results.items()):
            print(f"  Recall@{k}: {r:.4f}")

        print(f"\nThreshold classifier:")
        if thresh_oracle:
            print(f"  Oracle  — Acc: {thresh_oracle['accuracy']:.4f}, F1: {thresh_oracle['macro_f1']:.4f}")
        if thresh_e2e:
            print(f"  E2E     — Acc: {thresh_e2e['accuracy']:.4f}, F1: {thresh_e2e['macro_f1']:.4f}")

        print(f"\nMLP classifier:")
        if mlp_oracle:
            print(f"  Oracle  — Acc: {mlp_oracle['accuracy']:.4f}, F1: {mlp_oracle['macro_f1']:.4f}")
        if mlp_e2e:
            print(f"  E2E     — Acc: {mlp_e2e['accuracy']:.4f}, F1: {mlp_e2e['macro_f1']:.4f}")

        print(f"\nAll metrics saved to: {cfg.results.metrics_dir}")
        print(f"All figures saved to: {cfg.results.figures_dir}")

        # Log final results to tracker
        final = {}
        final["recall@5"] = recall_results.get(5, recall_results.get("5"))
        if thresh_oracle:
            final["threshold_oracle_f1"] = thresh_oracle["macro_f1"]
        if thresh_e2e:
            final["threshold_e2e_f1"] = thresh_e2e["macro_f1"]
        if mlp_oracle:
            final["mlp_oracle_f1"] = mlp_oracle["macro_f1"]
        if mlp_e2e:
            final["mlp_e2e_f1"] = mlp_e2e["macro_f1"]
        tracker.log_results(**final)
        tracker.complete()
        tracker.print_summary()

    except Exception as e:
        tracker.fail(str(e))
        raise


if __name__ == "__main__":
    main()
