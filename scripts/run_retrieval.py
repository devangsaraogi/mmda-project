"""Evaluate CLIP retrieval with Recall@K."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.models.clip_encoder import CLIPEncoder
from src.data.webqa_dataset import WebQADataset
from src.retrieval import EmbeddingIndex, CLIPRetriever, evaluate_retrieval
from src.evaluation import save_metrics
from src.visualization import plot_recall_at_k


def main():
    parser = argparse.ArgumentParser(description="Run CLIP retrieval evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to pre-computed image_index.pt from a previous run")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("retrieval_eval", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        # Load dataset
        dataset = WebQADataset(cfg)
        tracker.log_dataset(
            name="WebQA-Adv",
            total=len(dataset),
            train_size=len(dataset.get_split("train")),
            val_size=len(dataset.get_split("val")),
            test_size=len(dataset.get_split("test")),
            image_pool_size=len(dataset.get_all_image_ids()),
        )

        test_data = dataset.get_split("test")

        # Load encoder + index
        encoder = CLIPEncoder(cfg)

        index_path = args.embeddings or os.path.join(cfg.results.embeddings_dir, "image_index.pt")
        index = EmbeddingIndex()

        if os.path.exists(index_path):
            index.load(index_path)
        else:
            print("No pre-computed index found. Computing embeddings...")
            from src.retrieval import precompute_embeddings
            precompute_embeddings(encoder, dataset, index_path, batch_size=cfg.clip.batch_size)
            index.load(index_path)

        retriever = CLIPRetriever(encoder, index, cfg)

        # Evaluate
        recall_results = evaluate_retrieval(retriever, test_data, cfg.retrieval.top_k_values)

        # Save results
        save_metrics(
            {"recall_at_k": recall_results},
            os.path.join(cfg.results.metrics_dir, "retrieval_metrics.json"),
        )

        # Plot
        plot_recall_at_k(
            recall_results,
            os.path.join(cfg.results.figures_dir, "recall_at_k.png"),
        )

        tracker.log_results(**{f"recall@{k}": v for k, v in recall_results.items()})
        tracker.complete()

    except Exception as e:
        tracker.fail(str(e))
        raise

    print("Retrieval evaluation complete.")
    for k, r in sorted(recall_results.items()):
        print(f"  Recall@{k}: {r:.4f}")


if __name__ == "__main__":
    main()
