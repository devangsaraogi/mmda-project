"""Full multimodal pipeline: CLIP visual + BM25/NLI text + fusion."""

import argparse
import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.models.clip_encoder import CLIPEncoder
from src.data.webqa_dataset import WebQADataset
from src.retrieval import EmbeddingIndex, CLIPRetriever, evaluate_retrieval
from src.retrieval_text import BM25Retriever
from src.verification import (
    verify_oracle as visual_verify_oracle,
    verify_e2e as visual_verify_e2e,
)
from src.verification_text import (
    NLIVerifier,
    build_text_evidence_map_oracle,
    build_text_evidence_map_bm25,
)
from src.verification_multimodal import align_features_by_claim_id, MULTIMODAL_SETTINGS
from src.models.fusion_classifier import FusionMLP, GatedFusionClassifier
from src.evaluation import (
    compute_verification_metrics,
    per_type_breakdown,
    oracle_vs_e2e_comparison,
    save_metrics,
)
from src.visualization import plot_confusion_matrix, plot_per_type_breakdown

logger = logging.getLogger(__name__)


def train_and_evaluate_fusion(
    train_data, val_data, test_data, cfg, device, fusion_type="score", tag="",
):
    """Train a fusion model and evaluate on test set.

    Args:
        train/val/test_data: Aligned multimodal feature dicts.
        cfg: Config.
        device: torch device.
        fusion_type: "score" or "gated".
        tag: Label for logging.

    Returns:
        (test_metrics, per_type, model)
    """
    visual_dim = train_data["visual_features"].shape[1]
    text_dim = train_data["text_features"].shape[1]

    fusion_cfg = cfg.get("fusion", {})
    hidden_dims = list(fusion_cfg.get("hidden_dims", [256, 128]))
    class_weights_list = list(fusion_cfg.get("class_weights", cfg.verification.mlp.class_weights))
    lr = float(fusion_cfg.get("learning_rate", 0.001))
    batch_size = int(fusion_cfg.get("batch_size", 64))
    num_epochs = int(fusion_cfg.get("num_epochs", 50))
    patience_max = int(fusion_cfg.get("early_stopping_patience", 7))

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

    class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Prepare tensors
    Xv_tr = torch.tensor(train_data["visual_features"], dtype=torch.float32)
    Xt_tr = torch.tensor(train_data["text_features"], dtype=torch.float32)
    y_tr = torch.tensor(train_data["labels"], dtype=torch.long)
    Xv_val = torch.tensor(val_data["visual_features"], dtype=torch.float32)
    Xt_val = torch.tensor(val_data["text_features"], dtype=torch.float32)
    y_val = torch.tensor(val_data["labels"], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xv_tr, Xt_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xv_val, Xt_val, y_val), batch_size=batch_size)

    best_f1 = 0.0
    patience = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        for vb, tb, yb in train_loader:
            vb, tb, yb = vb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(vb, tb), yb).backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for vb, tb, yb in val_loader:
                vb, tb, yb = vb.to(device), tb.to(device), yb.to(device)
                preds.extend(model(vb, tb).argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        vf1 = f1_score(labels, preds, average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1 = vf1
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_max:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # Test evaluation
    Xv_te = torch.tensor(test_data["visual_features"], dtype=torch.float32).to(device)
    Xt_te = torch.tensor(test_data["text_features"], dtype=torch.float32).to(device)
    y_te = test_data["labels"]

    model.eval()
    with torch.no_grad():
        test_preds = model(Xv_te, Xt_te).argmax(1).cpu().numpy()

    metrics = compute_verification_metrics(y_te, test_preds)
    type_bd = per_type_breakdown(y_te, test_preds, test_data["misinfo_types"])

    logger.info("[%s] %s — Acc: %.4f, F1: %.4f (val_f1=%.4f)",
                fusion_type, tag, metrics["accuracy"], metrics["macro_f1"], best_f1)

    return metrics, type_bd, model


def main():
    parser = argparse.ArgumentParser(description="Full multimodal pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to pre-computed CLIP image_index.pt")
    parser.add_argument("--nli-scores", type=str, default=None,
                        help="Path to pre-computed NLI scores (.pt)")
    parser.add_argument("--debug", action="store_true",
                        help="Run on small subset")
    parser.add_argument("--per-claim-retrieval", action="store_true",
                        help="Use per-claim CLIP retrieval (dataset must carry image_candidate_ids).")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_dir = create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("multimodal_pipeline", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        # === Step 1: Load dataset ===
        print("\n" + "=" * 60)
        print("STEP 1: Loading dataset")
        print("=" * 60)

        dataset = WebQADataset(cfg)
        print(f"  Total claims: {len(dataset)}")
        print(f"  Image pool: {len(dataset.get_all_image_ids())}")

        tracker.log_dataset(
            name="WebQA-Adv",
            total=len(dataset),
            train_size=len(dataset.get_split("train")),
            val_size=len(dataset.get_split("val")),
            test_size=len(dataset.get_split("test")),
            image_pool_size=len(dataset.get_all_image_ids()),
        )

        # === Step 2: Load CLIP embeddings ===
        print("\n" + "=" * 60)
        print("STEP 2: Loading CLIP embeddings")
        print("=" * 60)

        encoder = CLIPEncoder(cfg)

        if args.embeddings and os.path.exists(args.embeddings):
            embedding_index = EmbeddingIndex()
            embedding_index.load(args.embeddings)
            print(f"  Loaded embeddings from {args.embeddings}")
        else:
            raise FileNotFoundError(
                "Pre-computed CLIP embeddings required. "
                "Run precompute_embeddings.py first or pass --embeddings."
            )

        # === Step 3: Visual features (all splits, oracle + E2E) ===
        print("\n" + "=" * 60)
        print("STEP 3: Extracting visual features")
        print("=" * 60)

        retriever = CLIPRetriever(encoder, embedding_index, cfg)
        default_top_k = cfg.retrieval.default_top_k

        per_claim_retrieval = bool(args.per_claim_retrieval) or bool(
            getattr(cfg.retrieval, "per_claim", False)
        )
        if per_claim_retrieval:
            print(f"  Visual E2E retrieval mode: PER-CLAIM (candidate pools from image_candidate_ids)")
        else:
            print(f"  Visual E2E retrieval mode: GLOBAL (full {len(dataset.get_all_image_ids())}-image pool)")

        visual_data = {}
        for split_name in ["train", "val", "test"]:
            split_data = dataset.get_split(split_name)

            oracle_result = visual_verify_oracle(encoder, split_data, embedding_index=embedding_index)
            visual_data[f"{split_name}_oracle"] = oracle_result

            e2e_result = visual_verify_e2e(
                encoder, retriever, split_data, default_top_k,
                embedding_index=embedding_index,
                per_claim=per_claim_retrieval,
            )
            visual_data[f"{split_name}_e2e"] = e2e_result

            print(f"  {split_name}: oracle={len(oracle_result['labels'])}, e2e={len(e2e_result['labels'])}")

        # Save visual features for fusion reuse
        torch.save({
            "oracle": {s: {k: v for k, v in visual_data[f"{s}_oracle"].items()}
                       for s in ["train", "val", "test"]},
            "e2e": {s: {k: v for k, v in visual_data[f"{s}_e2e"].items()}
                    for s in ["train", "val", "test"]},
        }, os.path.join(run_dir, "visual_features.pt"))

        tracker.log_step("visual_features", status="complete")

        # === Step 4: BM25 text retrieval ===
        print("\n" + "=" * 60)
        print("STEP 4: BM25 Text Retrieval")
        print("=" * 60)

        text_cfg = cfg.get("text", {})
        bm25_cfg = text_cfg.get("bm25", {})
        top_k_values = list(bm25_cfg.get("top_k_values", [1, 3, 5, 10]))
        text_top_k = int(bm25_cfg.get("default_top_k", 5))

        bm25 = BM25Retriever()
        bm25_recall = bm25.evaluate_recall(dataset, split="test", top_k_values=top_k_values)

        for k, r in sorted(bm25_recall.items()):
            print(f"  BM25 Recall@{k}: {r:.4f}")

        save_metrics(
            {"bm25_recall_at_k": bm25_recall},
            os.path.join(cfg.results.metrics_dir, "bm25_retrieval.json"),
        )
        tracker.log_step("bm25_retrieval", **{f"recall@{k}": v for k, v in bm25_recall.items()})

        # === Step 5: NLI text features (all splits, oracle + E2E) ===
        print("\n" + "=" * 60)
        print("STEP 5: NLI Text Features")
        print("=" * 60)

        nli_cfg = text_cfg.get("nli", {})
        nli_model = str(nli_cfg.get("model_name", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"))
        nli_bs = int(nli_cfg.get("batch_size", 32))

        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        if args.nli_scores and os.path.exists(args.nli_scores):
            print(f"  Loading pre-computed NLI scores from {args.nli_scores}")
            nli_cache = torch.load(args.nli_scores, map_location="cpu", weights_only=False)
            text_data = {}
            for mode in ["oracle", "e2e"]:
                for split_name in ["train", "val", "test"]:
                    text_data[f"{split_name}_{mode}"] = nli_cache[mode][split_name]
        else:
            nli = NLIVerifier(nli_model, device=device_str, batch_size=nli_bs)
            text_data = {}

            for split_name in ["train", "val", "test"]:
                # Oracle
                oracle_ev = build_text_evidence_map_oracle(dataset, split=split_name)
                oracle_res = nli.extract_features_batch(
                    dataset, oracle_ev, split=split_name, top_k=text_top_k,
                )
                text_data[f"{split_name}_oracle"] = oracle_res

                # E2E
                bm25_ev = build_text_evidence_map_bm25(bm25, dataset, split=split_name, top_k=text_top_k)
                e2e_res = nli.extract_features_batch(
                    dataset, bm25_ev, split=split_name, top_k=text_top_k,
                )
                text_data[f"{split_name}_e2e"] = e2e_res

                print(f"  {split_name}: oracle={len(oracle_res['labels'])}, e2e={len(e2e_res['labels'])}")

            # Cache for reuse
            torch.save({
                "oracle": {s: {k: v for k, v in text_data[f"{s}_oracle"].items() if k != "raw_scores"}
                           for s in ["train", "val", "test"]},
                "e2e": {s: {k: v for k, v in text_data[f"{s}_e2e"].items() if k != "raw_scores"}
                        for s in ["train", "val", "test"]},
            }, os.path.join(run_dir, "nli_scores.pt"))

        # Save text features for fusion reuse
        torch.save({
            "oracle": {s: {k: v for k, v in text_data[f"{s}_oracle"].items() if k != "raw_scores"}
                       for s in ["train", "val", "test"]},
            "e2e": {s: {k: v for k, v in text_data[f"{s}_e2e"].items() if k != "raw_scores"}
                    for s in ["train", "val", "test"]},
        }, os.path.join(run_dir, "text_features.pt"))

        tracker.log_step("text_features", status="complete")

        # === Step 6: Fusion training + evaluation ===
        print("\n" + "=" * 60)
        print("STEP 6: Multimodal Fusion")
        print("=" * 60)

        device = torch.device(device_str)
        fusion_cfg = cfg.get("fusion", {})
        fusion_types_str = str(fusion_cfg.get("type", "score"))
        # Support comma-separated fusion types, e.g. "score,gated"
        fusion_types = [t.strip() for t in fusion_types_str.split(",")]

        all_metrics = {}

        # Also evaluate visual-only and text-only MLP baselines for comparison
        print("\n--- Single-modality baselines (MLP) ---")
        for modality, data_dict, mode_list in [
            ("visual", visual_data, ["oracle", "e2e"]),
            ("text", text_data, ["oracle", "e2e"]),
        ]:
            for mode in mode_list:
                key = f"{modality}_{mode}"
                test_feats = data_dict[f"test_{mode}"]
                train_feats = data_dict[f"train_{mode}"]
                val_feats = data_dict[f"val_{mode}"]

                X_tr = torch.tensor(train_feats["features"], dtype=torch.float32).to(device)
                y_tr = torch.tensor(train_feats["labels"], dtype=torch.long).to(device)
                X_val = torch.tensor(val_feats["features"], dtype=torch.float32).to(device)
                y_val = torch.tensor(val_feats["labels"], dtype=torch.long).to(device)
                X_te = torch.tensor(test_feats["features"], dtype=torch.float32).to(device)
                y_te = test_feats["labels"]

                # Quick MLP
                dim = X_tr.shape[1]
                mlp = nn.Sequential(
                    nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 3),
                ).to(device)
                cw = torch.tensor(list(cfg.verification.mlp.class_weights), dtype=torch.float32).to(device)
                crit = nn.CrossEntropyLoss(weight=cw)
                opt = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)

                loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
                best_f1, best_st, pat = 0, None, 0
                for ep in range(50):
                    mlp.train()
                    for xb, yb in loader:
                        opt.zero_grad()
                        crit(mlp(xb), yb).backward()
                        opt.step()
                    mlp.eval()
                    with torch.no_grad():
                        vp = mlp(X_val).argmax(1).cpu().numpy()
                    vf = f1_score(y_val.cpu().numpy(), vp, average="macro", zero_division=0)
                    if vf > best_f1:
                        best_f1 = vf
                        best_st = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                        pat = 0
                    else:
                        pat += 1
                        if pat >= 7:
                            break

                if best_st:
                    mlp.load_state_dict(best_st)
                    mlp.to(device)
                mlp.eval()
                with torch.no_grad():
                    tp = mlp(X_te).argmax(1).cpu().numpy()
                m = compute_verification_metrics(y_te if isinstance(y_te, np.ndarray) else np.array(y_te), tp)
                all_metrics[key] = m
                print(f"  {key} — Acc: {m['accuracy']:.4f}, F1: {m['macro_f1']:.4f}")

        # Multimodal fusion — iterate over all fusion types
        for fusion_type in fusion_types:
            print(f"\n--- Multimodal fusion ({fusion_type}) ---")
            for setting_name, vis_mode, txt_mode in MULTIMODAL_SETTINGS:
                train_aligned = align_features_by_claim_id(
                    visual_data[f"train_{vis_mode}"], text_data[f"train_{txt_mode}"]
                )
                val_aligned = align_features_by_claim_id(
                    visual_data[f"val_{vis_mode}"], text_data[f"val_{txt_mode}"]
                )
                test_aligned = align_features_by_claim_id(
                    visual_data[f"test_{vis_mode}"], text_data[f"test_{txt_mode}"]
                )

                metrics, type_bd, model = train_and_evaluate_fusion(
                    train_aligned, val_aligned, test_aligned,
                    cfg, device, fusion_type=fusion_type, tag=setting_name,
                )

                key_prefix = f"fusion_{fusion_type}" if len(fusion_types) > 1 else "fusion"
                metric_key = f"{key_prefix}_{setting_name}"
                all_metrics[metric_key] = metrics
                all_metrics[f"{metric_key}_per_type"] = type_bd

                print(f"  {setting_name} — Acc: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}")

                # Save per-setting metrics + visualizations
                save_metrics(metrics, os.path.join(cfg.results.metrics_dir, f"{metric_key}.json"))
                plot_confusion_matrix(
                    metrics["confusion_matrix"], metrics["confusion_labels"],
                    os.path.join(cfg.results.figures_dir, f"{metric_key}_cm.png"),
                    title=f"Fusion ({fusion_type}): {setting_name}",
                )
                plot_per_type_breakdown(
                    type_bd,
                    os.path.join(cfg.results.figures_dir, f"{metric_key}_per_type.png"),
                )

                # Save checkpoint
                ckpt_path = os.path.join(cfg.results.checkpoints_dir, f"{metric_key}.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

        # === Summary ===
        print("\n" + "=" * 60)
        print("MULTIMODAL PIPELINE COMPLETE — Summary")
        print("=" * 60)

        print("\n  Comparison Table:")
        print(f"  {'Model':<40} {'Acc':>8} {'F1':>8}")
        print("  " + "-" * 56)
        for key in sorted(all_metrics.keys()):
            if "_per_type" not in key:
                m = all_metrics[key]
                print(f"  {key:<40} {m['accuracy']:>8.4f} {m['macro_f1']:>8.4f}")

        # Save all metrics
        save_metrics(
            {k: v for k, v in all_metrics.items() if "_per_type" not in k},
            os.path.join(cfg.results.metrics_dir, "all_metrics_summary.json"),
        )

        # Tracker
        final = {}
        for key, m in all_metrics.items():
            if "_per_type" not in key:
                final[f"{key}_f1"] = m["macro_f1"]
        if bm25_recall:
            final["bm25_recall@5"] = bm25_recall.get(5, 0)
        tracker.log_results(**final)
        tracker.complete()
        tracker.print_summary()

    except Exception as e:
        tracker.fail(str(e))
        raise


if __name__ == "__main__":
    main()
