"""Small utilities shared across the project."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible classroom experiments."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch is optional for this project.
        pass


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a pandas DataFrame as CSV and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> Path:
    """Save a JSON file with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def notebook_friendly_path(path: str | Path) -> str:
    """Return a path string that is pleasant to print inside notebooks."""
    return os.fspath(Path(path))


def print_section(title: str) -> None:
    """Pretty section print for notebooks."""
    bar = "=" * max(40, len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")
