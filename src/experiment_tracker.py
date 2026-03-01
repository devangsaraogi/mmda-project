"""Experiment tracking utility.

Logs comprehensive experiment metadata including system/GPU info,
config snapshot, dataset stats, per-step results, and training epochs.
Saves everything to a JSON file for reproducibility and inspection.
"""

import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf


class ExperimentTracker:
    """Track and log experiment details for reproducibility."""

    def __init__(self, experiment_name=None, log_dir="results/experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = datetime.now()
        self.start_timestamp = time.time()

        if experiment_name is None:
            experiment_name = f"exp_{self.start_time.strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.json"

        self.data = {
            "experiment_name": experiment_name,
            "start_time": self.start_time.isoformat(),
            "status": "running",
        }

        self._capture_system_info()

    # ------------------------------------------------------------------
    # System info
    # ------------------------------------------------------------------

    def _capture_system_info(self):
        """Capture system, GPU, and LSF job information."""
        self.data["system"] = {
            "hostname": socket.gethostname(),
            "job_id": os.environ.get("LSB_JOBID", "N/A"),
            "job_name": os.environ.get("LSB_JOBNAME", "N/A"),
            "queue": os.environ.get("LSB_QUEUE", "N/A"),
            "user": os.environ.get("USER", "N/A"),
        }

        if torch.cuda.is_available():
            self.data["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 2
                ),
            }
        else:
            self.data["gpu"] = {"available": False}

        self.data["pytorch"] = {
            "version": torch.__version__,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.cuda.is_available() else None
            ),
        }

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_config(self, cfg):
        """Snapshot the full OmegaConf config as a dict."""
        self.data["config"] = OmegaConf.to_container(cfg, resolve=True)
        self._save()

    def log_dataset(self, name, total, train_size, val_size, test_size,
                    image_pool_size, **kwargs):
        """Log dataset statistics."""
        self.data["dataset"] = {
            "name": name,
            "total_claims": total,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "image_pool_size": image_pool_size,
            **kwargs,
        }
        self._save()

    def log_step(self, step_name, **metrics):
        """Log results for a named pipeline step (e.g. 'retrieval', 'verification')."""
        if "steps" not in self.data:
            self.data["steps"] = {}
        self.data["steps"][step_name] = {
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self._save()

    def log_epoch(self, epoch, train_loss, train_metric, val_loss, val_metric,
                  learning_rate=None, epoch_time=None):
        """Log a single training epoch."""
        if "epochs" not in self.data:
            self.data["epochs"] = []

        self.data["epochs"].append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_f1": float(train_metric),
            "val_loss": float(val_loss),
            "val_f1": float(val_metric),
            "learning_rate": float(learning_rate) if learning_rate else None,
            "epoch_time": float(epoch_time) if epoch_time else None,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def log_results(self, **kwargs):
        """Log final results."""
        self.data["results"] = {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in kwargs.items()
        }
        self._save()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def complete(self, status="completed"):
        """Mark experiment as complete and record duration."""
        end_time = datetime.now()
        duration = time.time() - self.start_timestamp

        self.data["status"] = status
        self.data["end_time"] = end_time.isoformat()
        self.data["duration_seconds"] = round(duration, 1)
        self.data["duration_formatted"] = self._format_duration(duration)
        self._save()

        print(f"\n{'=' * 60}")
        print(f"Experiment '{self.experiment_name}' {status}")
        print(f"Duration: {self.data['duration_formatted']}")
        print(f"Log saved to: {self.log_file}")
        print(f"{'=' * 60}\n")

    def fail(self, error_msg):
        """Mark experiment as failed."""
        self.data["error"] = str(error_msg)
        self.complete(status="failed")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_duration(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def print_summary(self):
        """Print a short summary to stdout."""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'=' * 60}")

        sys = self.data.get("system", {})
        print(f"  Host: {sys.get('hostname', '?')}  Job: {sys.get('job_id', '?')}")

        gpu = self.data.get("gpu", {})
        if gpu.get("available"):
            print(f"  GPU: {gpu['device_name']} ({gpu['total_memory_gb']} GB)")

        ds = self.data.get("dataset", {})
        if ds:
            print(f"  Dataset: {ds.get('name', '?')} — {ds.get('total_claims', '?')} claims, {ds.get('image_pool_size', '?')} images")

        results = self.data.get("results", {})
        if results:
            print("  Results:")
            for k, v in results.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

        print(f"{'=' * 60}\n")
