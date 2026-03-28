"""Train fusion classifier on pre-computed visual + text features."""

import argparse
import os
import sys
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.models.fusion_classifier import FusionMLP, GatedFusionClassifier
from src.verification_multimodal import align_features_by_claim_id
from src.evaluation import save_metrics

logger = logging.getLogger(__name__)


def load_precomputed_features(visual_path: str, text_path: str, split: str) -> dict:
    """Load and align pre-computed visual + text features for a split."""
    visual_data = torch.load(visual_path, map_location="cpu", weights_only=False)
    text_data = torch.load(text_path, map_location="cpu", weights_only=False)

    # Handle nested structure (e.g. {train: {...}, val: {...}, test: {...}})
    if split in visual_data:
        visual_data = visual_data[split]
    if split in text_data:
        text_data = text_data[split]

    return align_features_by_claim_id(visual_data, text_data)


def train_fusion(cfg, visual_features_path: str, text_features_path: str,
                 visual_mode: str = "oracle", text_mode: str = "oracle",
                 tracker=None):
    """Train a fusion classifier.

    Args:
        cfg: Config.
        visual_features_path: Path to .pt with visual features per split.
        text_features_path: Path to .pt with text features per split.
        visual_mode: "oracle" or "e2e".
        text_mode: "oracle" or "e2e".
        tracker: ExperimentTracker.

    Returns:
        (model, best_val_f1, checkpoint_path)
    """
    ensure_dirs(cfg)

    # Load aligned features
    logger.info("Loading features: visual=%s, text=%s", visual_mode, text_mode)
    train_data = load_precomputed_features(visual_features_path, text_features_path, "train")
    val_data = load_precomputed_features(visual_features_path, text_features_path, "val")

    visual_dim = train_data["visual_features"].shape[1]
    text_dim = train_data["text_features"].shape[1]

    X_train_v = torch.tensor(train_data["visual_features"], dtype=torch.float32)
    X_train_t = torch.tensor(train_data["text_features"], dtype=torch.float32)
    y_train = torch.tensor(train_data["labels"], dtype=torch.long)

    X_val_v = torch.tensor(val_data["visual_features"], dtype=torch.float32)
    X_val_t = torch.tensor(val_data["text_features"], dtype=torch.float32)
    y_val = torch.tensor(val_data["labels"], dtype=torch.long)

    logger.info("Train: %d samples, visual_dim=%d, text_dim=%d", len(y_train), visual_dim, text_dim)
    logger.info("Val: %d samples", len(y_val))
    logger.info("Label distribution (train): %s", np.bincount(y_train.numpy(), minlength=3))

    # Select model
    fusion_cfg = cfg.get("fusion", {})
    fusion_type = str(fusion_cfg.get("type", "score"))
    hidden_dims = list(fusion_cfg.get("hidden_dims", [256, 128]))
    lr = float(fusion_cfg.get("learning_rate", 0.001))
    batch_size = int(fusion_cfg.get("batch_size", 64))
    num_epochs = int(fusion_cfg.get("num_epochs", 50))
    patience_max = int(fusion_cfg.get("early_stopping_patience", 7))
    class_weights_list = list(fusion_cfg.get("class_weights", cfg.verification.mlp.class_weights))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fusion_type == "gated":
        model = GatedFusionClassifier(
            visual_dim=visual_dim, text_dim=text_dim,
            hidden_dim=hidden_dims[0], num_classes=3,
        ).to(device)
    else:
        model = FusionMLP(
            visual_dim=visual_dim, text_dim=text_dim,
            hidden_dims=hidden_dims, num_classes=3,
        ).to(device)

    logger.info("Fusion model (%s):\n%s", fusion_type, model)

    class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_v, X_train_t, y_train),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_v, X_val_t, y_val),
        batch_size=batch_size,
    )

    # TensorBoard
    tag = f"fusion_{visual_mode}_{text_mode}"
    writer = SummaryWriter(log_dir=os.path.join(cfg.logging.tensorboard_dir, tag))

    best_val_f1 = 0.0
    patience = 0
    best_state = None

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        for vb, tb, yb in train_loader:
            vb, tb, yb = vb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(vb, tb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)
            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_labels.extend(yb.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for vb, tb, yb in val_loader:
                vb, tb, yb = vb.to(device), tb.to(device), yb.to(device)
                logits = model(vb, tb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(yb)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        epoch_time = time.time() - epoch_start

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("macro_f1", {"train": train_f1, "val": val_f1}, epoch)

        if tracker:
            tracker.log_epoch(epoch + 1, train_loss, train_f1, val_loss, val_f1,
                              epoch_time=epoch_time)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f train_f1=%.4f val_loss=%.4f val_f1=%.4f (%.1fs)",
            epoch + 1, num_epochs, train_loss, train_f1, val_loss, val_f1, epoch_time,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  New best (val_f1=%.4f)", val_f1)
        else:
            patience += 1
            if patience >= patience_max:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    writer.close()

    # Restore best and save
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    ckpt_name = f"fusion_{fusion_type}_{visual_mode}_{text_mode}.pt"
    ckpt_path = os.path.join(cfg.results.checkpoints_dir, ckpt_name)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    logger.info("Saved fusion checkpoint to %s", ckpt_path)

    save_metrics(
        {
            "fusion_type": fusion_type,
            "visual_mode": visual_mode,
            "text_mode": text_mode,
            "best_val_f1": float(best_val_f1),
            "visual_dim": visual_dim,
            "text_dim": text_dim,
        },
        os.path.join(cfg.results.metrics_dir, f"fusion_training_{visual_mode}_{text_mode}.json"),
    )

    return model, best_val_f1, ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train fusion classifier")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--visual-features", type=str, required=True,
                        help="Path to visual features .pt file")
    parser.add_argument("--text-features", type=str, required=True,
                        help="Path to text features .pt file")
    parser.add_argument("--visual-mode", type=str, default="oracle",
                        choices=["oracle", "e2e"])
    parser.add_argument("--text-mode", type=str, default="oracle",
                        choices=["oracle", "e2e"])
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)

    tag = f"fusion_{args.visual_mode}_{args.text_mode}"
    tracker = ExperimentTracker(f"train_{tag}", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        model, best_f1, ckpt = train_fusion(
            cfg, args.visual_features, args.text_features,
            visual_mode=args.visual_mode, text_mode=args.text_mode,
            tracker=tracker,
        )
        tracker.log_results(best_val_f1=best_f1, checkpoint=ckpt)
        tracker.complete()
    except Exception as e:
        tracker.fail(str(e))
        raise


if __name__ == "__main__":
    main()
