# -*- coding: utf-8 -*-
"""
Train an MLP classifier for Minecraft player types.
- Auto-detect feature columns from data/samples.csv
- Group-based split by player_id to avoid leakage
- Class-weighted cross entropy, AdamW + cosine annealing
- Early stopping, AMP (optional)
- Saves best.pt with feat_cols & classes for later ONNX export
"""

import os
import math
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------
# 超參數（可視需要調整；若有 config.yaml 也可自行改成讀 yaml）
# --------------------------
RANDOM_SEED = 42
DATA_PATH = Path("data/samples.csv")  # 由 build_samples.py 產生
LABEL_COL = "label"
GROUP_COL = "player_id"

BATCH_SIZE = 512
NUM_EPOCHS = 80
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 4
MIXED_PRECISION = True  # 若沒有 GPU 自動無效

HIDDEN_SIZES = (256, 128, 64)
DROPOUT = 0.15
ACTIVATION = "relu"  # relu/gelu/elu

LR = 1.5e-3
WEIGHT_DECAY = 1e-4
USE_COSINE = True
COSINE_TMAX = 40

BEST_CKPT = "best.pt"

# --------------------------
# 小工具
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_sizes=(256,128,64), dropout=0.1, activation="relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}
        act = acts.get(activation.lower(), nn.ReLU)
        layers = []
        d = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), act(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits

def auto_feature_columns(df: pd.DataFrame) -> list[str]:
    """抓所有 rate_* 與 dist_*；若存在也納入 _cluster/pmax/min_dist/dist_to_assigned/AFK_ratio/active_minutes。"""
    feats = [c for c in df.columns if c.startswith("rate_") or c.startswith("dist_")]
    for extra in ["_cluster", "pmax", "min_dist", "dist_to_assigned", "AFK_ratio", "active_minutes"]:
        if extra in df.columns and extra not in feats:
            feats.append(extra)
    # 排除非特徵
    for non in ["server", "player_id", "row_idx", "label", "time"]:
        if non in feats:
            feats.remove(non)
    if not feats:
        raise RuntimeError("找不到可用特徵欄位（rate_* / dist_* 等）。請檢查 data/samples.csv。")
    return feats

def group_train_val_test_split(df: pd.DataFrame, group_col: str, test_ratio=0.2, val_ratio_of_train=0.15, seed=42):
    """先依 group 切出 test；再在 train 部分切出 val（可用 stratify）。"""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    groups = df[group_col].astype(str)
    idx_tr, idx_te = next(gss.split(df, groups=groups))
    df_tr_full = df.iloc[idx_tr].copy()
    df_te = df.iloc[idx_te].copy()

    # 再從訓練集切出驗證集（用標籤做 stratify 以保持比例）
    df_tr, df_va = train_test_split(
        df_tr_full,
        test_size=val_ratio_of_train,
        random_state=seed,
        stratify=df_tr_full[LABEL_COL] if df_tr_full[LABEL_COL].nunique() > 1 else None,
    )
    return df_tr, df_va, df_te

def build_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, num_workers):
    ds_tr = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    ds_va = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    ds_te = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_tr, dl_va, dl_te

# --------------------------
# 主流程
# --------------------------
def main():
    set_seed(RANDOM_SEED)

    # 讀資料
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 {DATA_PATH}，請先執行 src/build_samples.py")
    df = pd.read_csv(DATA_PATH)

    # 檢查必要欄
    for col in [LABEL_COL, GROUP_COL]:
        if col not in df.columns:
            raise KeyError(f"samples.csv 缺少必要欄位：{col}")

    # 自動抓特徵欄位
    feat_cols = auto_feature_columns(df)

    # Label 編碼
    le = LabelEncoder()
    y_all = le.fit_transform(df[LABEL_COL].astype(str))
    classes = list(le.classes_)

    # Group 切分
    df_tr, df_va, df_te = group_train_val_test_split(
        df, group_col=GROUP_COL, test_ratio=0.2, val_ratio_of_train=0.15, seed=RANDOM_SEED
    )

    def xy(d: pd.DataFrame):
        X = d[feat_cols].to_numpy(np.float32)
        y = le.transform(d[LABEL_COL].astype(str))
        return X, y

    X_tr, y_tr = xy(df_tr)
    X_va, y_va = xy(df_va)
    X_te, y_te = xy(df_te)

    dl_tr, dl_va, dl_te = build_loaders(
        X_tr, y_tr, X_va, y_va, X_te, y_te, BATCH_SIZE, NUM_WORKERS
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(
        in_dim=len(feat_cols),
        num_classes=len(classes),
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
        activation=ACTIVATION,
    ).to(device)

    # 類別權重（balanced）
    _, counts = np.unique(y_all, return_counts=True)
    class_w = (counts.sum() / (len(counts) * counts)).astype(np.float32)  # 1/freq 正規化
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=COSINE_TMAX) if USE_COSINE else None
    scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PRECISION and torch.cuda.is_available()))

    best_val = -1.0
    bad = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sch: sch.step()

            tr_loss_sum += loss.item() * yb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += yb.size(0)

        train_acc = tr_correct / tr_total
        train_loss = tr_loss_sum / tr_total

        # ---- Valid ----
        model.eval()
        va_loss_sum, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss_sum += loss.item() * yb.size(0)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total += yb.size(0)

        val_acc = va_correct / va_total
        val_loss = va_loss_sum / va_total

        print(f"[E{epoch:03d}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"loss(tr/va)={train_loss:.4f}/{val_loss:.4f}")

        # Early stopping on val_acc
        if val_acc > best_val:
            best_val = val_acc
            bad = 0
            torch.save({"model": model.state_dict(),
                        "classes": classes,
                        "feat_cols": feat_cols}, BEST_CKPT)
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print("Early stopping.")
                break

    # ---- Test ----
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_y.append(yb.numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    preds = all_logits.argmax(1)

    print("\n=== Test report ===")
    print(classification_report(all_y, preds, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(all_y, preds))
    print(f"Best val acc = {best_val:.4f}")
    print(f"Saved checkpoint -> {BEST_CKPT}")
    print(f"Feature columns ({len(feat_cols)}): {feat_cols[:8]}{'...' if len(feat_cols)>8 else ''}")

if __name__ == "__main__":
    main()
