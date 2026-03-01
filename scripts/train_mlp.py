"""Train MLP classifier on CLIP features for verdict prediction."""

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

from src.config import load_config, set_seed, setup_logging, ensure_dirs
from src.experiment_tracker import ExperimentTracker
from src.models.clip_encoder import CLIPEncoder
from src.models.mlp_classifier import MLPClassifier
from src.data.webqa_dataset import WebQADataset
from src.verification import verify_oracle, build_evidence_map_oracle, extract_features
from src.evaluation import save_metrics

logger = logging.getLogger(__name__)


def train_mlp(cfg, tracker=None):
    ensure_dirs(cfg)

    # Load dataset
    dataset = WebQADataset(cfg)

    train_data = dataset.get_split("train")
    val_data = dataset.get_split("val")

    if tracker:
        tracker.log_dataset(
            name="WebQA-Adv",
            total=len(dataset),
            train_size=len(train_data),
            val_size=len(val_data),
            test_size=len(dataset.get_split("test")),
            image_pool_size=len(dataset.get_all_image_ids()),
        )

    # Encode features
    encoder = CLIPEncoder(cfg)

    logger.info("Extracting training features...")
    train_evidence = build_evidence_map_oracle(train_data)
    train_feats = extract_features(encoder, train_data, train_evidence)

    logger.info("Extracting validation features...")
    val_evidence = build_evidence_map_oracle(val_data)
    val_feats = extract_features(encoder, val_data, val_evidence)

    # Select input mode
    input_mode = cfg.verification.mlp.input_mode
    if input_mode == "features":
        X_train = torch.tensor(train_feats["features"], dtype=torch.float32)
        X_val = torch.tensor(val_feats["features"], dtype=torch.float32)
    else:
        X_train = torch.tensor(train_feats["embeddings"], dtype=torch.float32)
        X_val = torch.tensor(val_feats["embeddings"], dtype=torch.float32)

    y_train = torch.tensor(train_feats["labels"], dtype=torch.long)
    y_val = torch.tensor(val_feats["labels"], dtype=torch.long)

    logger.info(f"Training set: {len(X_train)} samples, input_dim={X_train.shape[1]}")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Label distribution (train): {np.bincount(y_train.numpy(), minlength=3)}")

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.verification.mlp.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=cfg.verification.mlp.batch_size,
    )

    # Model
    device = torch.device(cfg.clip.device if torch.cuda.is_available() or cfg.clip.device == "cpu" else "cpu")
    model = MLPClassifier(cfg).to(device)
    logger.info(f"MLP architecture:\n{model}")

    # Class-weighted loss
    class_weights = torch.tensor(cfg.verification.mlp.class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.verification.mlp.learning_rate,
        weight_decay=cfg.verification.mlp.weight_decay,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg.logging.tensorboard_dir)

    # Training loop with early stopping
    best_val_f1 = 0.0
    patience_counter = 0
    best_checkpoint_path = os.path.join(cfg.results.checkpoints_dir, "mlp_best.pt")

    for epoch in range(cfg.verification.mlp.num_epochs):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        epoch_time = time.time() - epoch_start

        # Log
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("macro_f1", {"train": train_f1, "val": val_f1}, epoch)

        if tracker:
            lr = optimizer.param_groups[0]["lr"]
            tracker.log_epoch(epoch + 1, train_loss, train_f1, val_loss, val_f1,
                              learning_rate=lr, epoch_time=epoch_time)

        logger.info(
            f"Epoch {epoch+1}/{cfg.verification.mlp.num_epochs} — "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
            f"({epoch_time:.1f}s)"
        )

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"  New best model saved (val_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.verification.mlp.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    writer.close()

    # Save training summary
    save_metrics(
        {
            "best_val_f1": float(best_val_f1),
            "final_epoch": epoch + 1,
            "input_mode": input_mode,
            "checkpoint": best_checkpoint_path,
        },
        os.path.join(cfg.results.metrics_dir, "mlp_training.json"),
    )

    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f}")
    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train MLP classifier")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)

    tracker = ExperimentTracker("train_mlp", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        train_mlp(cfg, tracker=tracker)
        tracker.complete()
    except Exception as e:
        tracker.fail(str(e))
        raise


if __name__ == "__main__":
    main()
