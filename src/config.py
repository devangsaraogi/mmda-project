import os
import random
import logging

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str = None, overrides: list[str] = None) -> DictConfig:
    """Load YAML config with optional overrides.

    Args:
        config_path: Path to YAML config file. Defaults to configs/default.yaml.
        overrides: List of dotlist overrides, e.g. ["clip.device=cpu", "seed=123"].

    Returns:
        Merged OmegaConf DictConfig.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(project_root, "configs", "default.yaml")

    base_cfg = OmegaConf.load(default_path)

    if config_path and config_path != default_path:
        override_cfg = OmegaConf.load(config_path)
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    return base_cfg


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(level: str = "INFO", log_dir: str = "results/logs") -> None:
    """Configure root logger to write to both console and a log file.

    Each run creates a timestamped log file in *log_dir* so that logs
    persist for future inspection.
    """
    from datetime import datetime

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    log_level = getattr(logging, level.upper())
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    root.info("Logging to %s", log_file)


def ensure_dirs(cfg: DictConfig) -> None:
    """Create output directories from config."""
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    for key in ["figures_dir", "metrics_dir", "embeddings_dir", "checkpoints_dir", "experiments_dir"]:
        path = cfg.results.get(key)
        if path:
            os.makedirs(path, exist_ok=True)
