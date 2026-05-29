"""PyTorch MLP classifier used for latent representation analysis."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .models import evaluate_predictions


@dataclass
class MLPTrainingResult:
    model: object
    history: pd.DataFrame
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: Dict[str, float]
    device: str


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, DataLoader, TensorDataset
    except Exception as exc:
        raise ImportError("PyTorch is required for the MLP part. Install with: pip install torch") from exc


def seed_torch(seed: int = 42) -> None:
    torch, _, _, _ = require_torch()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_mlp_class(input_dim: int, hidden_dim: int = 256, latent_dim: int = 64, dropout: float = 0.25):
    """Create an MLP class dynamically after torch import."""
    torch, nn, _, _ = require_torch()

    class MLPClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, output_dim: int, dropout: float = 0.25):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(latent_dim, output_dim)

        def forward(self, x):
            z = self.feature_extractor(x)
            return self.classifier(z)

        def encode(self, x):
            return self.feature_extractor(x)

    return MLPClassifier


def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    dropout: float = 0.25,
    epochs: int = 120,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_size: float = 0.15,
    patience: int = 15,
    random_state: int = 42,
    verbose: bool = True,
) -> MLPTrainingResult:
    """Train an MLP classifier and evaluate it on the test set."""
    torch, nn, DataLoader, TensorDataset = require_torch()
    seed_torch(random_state)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = make_mlp_class(X_train.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout)
    model = model_cls(X_train.shape[1], hidden_dim, latent_dim, n_classes, dropout).to(device)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    val_x = torch.tensor(X_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.long, device=device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            n_seen += len(xb)
        train_loss = running_loss / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()
            val_pred = val_logits.argmax(dim=1).detach().cpu().numpy()
            val_acc = (val_pred == y_val).mean()

        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"[early stopping] No validation loss improvement for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_proba = predict_proba_mlp(model, X_test, device=device)
    y_pred = y_proba.argmax(axis=1)
    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    history = pd.DataFrame(history_rows)
    return MLPTrainingResult(model=model, history=history, y_pred=y_pred, y_proba=y_proba, metrics=metrics, device=device)


def predict_proba_mlp(model, X: np.ndarray, device: str | None = None, batch_size: int = 512) -> np.ndarray:
    """Predict class probabilities from a trained MLP."""
    torch, _, DataLoader, TensorDataset = require_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.vstack(probs)


def encode_mlp_latent(model, X: np.ndarray, device: str | None = None, batch_size: int = 512) -> np.ndarray:
    """Return latent features from the MLP encoder."""
    torch, _, DataLoader, TensorDataset = require_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    latents = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            z = model.encode(xb)
            latents.append(z.detach().cpu().numpy())
    return np.vstack(latents)
