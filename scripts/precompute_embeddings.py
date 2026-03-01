"""Pre-compute CLIP embeddings for all images in the dataset."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs
from src.experiment_tracker import ExperimentTracker
from src.models.clip_encoder import CLIPEncoder
from src.data.webqa_dataset import WebQADataset
from src.retrieval import precompute_embeddings


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CLIP image embeddings")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("overrides", nargs="*", help="Config overrides (dotlist)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("precompute_embeddings", cfg.results.experiments_dir)
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

        # Load CLIP encoder
        encoder = CLIPEncoder(cfg)

        # Pre-compute and save
        save_path = os.path.join(cfg.results.embeddings_dir, "image_index.pt")
        precompute_embeddings(encoder, dataset, save_path, batch_size=cfg.clip.batch_size)

        tracker.log_results(
            embeddings_path=save_path,
            num_images=len(dataset.get_all_image_ids()),
        )
        tracker.complete()

    except Exception as e:
        tracker.fail(str(e))
        raise

    print(f"Done. Embeddings saved to {save_path}")


if __name__ == "__main__":
    main()
