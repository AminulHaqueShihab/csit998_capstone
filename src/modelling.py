"""Shared modelling code: data loading, metrics, LSTM, XGBoost helper, bootstrap.

All models see identical train / validation / test dates (built by src.make_splits). Early stopping and
hyperparameter choice use the VALIDATION split only; the test split is touched once, in src.evaluate_final.
"""
import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import nn

from . import config, features
from .config import PipelineError

MODEL_CONFIG_FILE = config.ROOT / "results" / "frozen_config.json"
THRESHOLD = 0.5  # fixed decision threshold: never tuned on any split


# ------------------------------------------------------------------ data

def load_sequences():
    config.require_file(config.SEQUENCES, "modelling", "Run `python -m src.make_splits` first.")
    return np.load(config.SEQUENCES, allow_pickle=False)


def load_tabular(split: str, with_sentiment: bool):
    df = pd.read_csv(config.SPLITS_DIR / f"{split}.csv", parse_dates=["date"])
    cols = features.MODEL_PRICE_FEATURES + (features.MODEL_SENTIMENT_FEATURES if with_sentiment else [])
    return df[cols].to_numpy(dtype=float), df["target_direction"].to_numpy(dtype=int), df["date"]


# ------------------------------------------------------------------ metrics

def direction_metrics(y_true, proba, threshold: float = THRESHOLD) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def log_loss_value(y_true, proba, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(proba, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def paired_bootstrap_diff(y_true, proba_a, proba_b, metric="accuracy", n_boot=2000, seed=0) -> dict:
    """95% interval for metric(B) - metric(A) from resampling test days with replacement (paired)."""
    y = np.asarray(y_true)
    pa = (np.asarray(proba_a) >= THRESHOLD).astype(int)
    pb = (np.asarray(proba_b) >= THRESHOLD).astype(int)
    fn = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score}[metric]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = fn(y[idx], pb[idx]) - fn(y[idx], pa[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": float(fn(y, pb) - fn(y, pa)), "ci95_low": float(lo), "ci95_high": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "n_boot": n_boot}


# ------------------------------------------------------------------ LSTM

class DirectionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(self.drop(h[:, -1])).squeeze(-1)  # logit for P(next day up)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()


def train_lstm(Xtr, ytr, Xva, yva, params: dict, seed: int):
    """Train with Adam + early stopping on validation loss. Same budget for every model (ML-05)."""
    set_seed(seed)
    torch.set_num_threads(1)
    model = DirectionLSTM(Xtr.shape[2], params["hidden"], params["layers"], params["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 1e-4))
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr, dtype=torch.float32)
    xv, yv = torch.as_tensor(Xva, dtype=torch.float32), torch.as_tensor(yva, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    for epoch in range(1, params["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), params["batch_size"]):
            idx = order[i: i + params["batch_size"]]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        if val_loss < best["loss"] - 1e-5:
            best = {"loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= params["patience"]:
            break
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_val_loss": best["loss"], "epochs_run": epoch}


# ------------------------------------------------------------------ XGBoost

def train_xgb(Xtr, ytr, Xva, yva, params: dict, seed: int):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 500), learning_rate=params["learning_rate"],
        max_depth=params["max_depth"], min_child_weight=params.get("min_child_weight", 5),
        subsample=0.8, colsample_bytree=0.8, reg_lambda=params.get("reg_lambda", 5.0),
        eval_metric="logloss", early_stopping_rounds=30, tree_method="hist", n_jobs=1,
        random_state=seed)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model, {"best_iteration": int(model.best_iteration), "best_val_loss": float(model.best_score)}


def naive_majority(ytr) -> int:
    """ML-01: always predict the majority class of the TRAINING labels."""
    return int(np.mean(ytr) >= 0.5)
