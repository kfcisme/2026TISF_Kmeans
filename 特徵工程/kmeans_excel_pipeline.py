
import argparse
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


RAW_COLS = [
    "pickup","block_break","tnt_prime","multi_place","chat","block_damage","block_place",
    "craft","dmg_by_entity","death","explosion","furnace_extract","inv_close","inv_open",
    "bucket_empty","bucket_fill","cmd_pre","cmd_send","player_death","item_drop",
    "exp_change","interact","level_change","quit","respawn","teleport","chunkload",
    "redstone","afktime"
]

OPT_TIME_COLS = ["timestamp", "time", "ts", "datetime"]

def read_one_table(path: Path) -> pd.DataFrame:
    """Read one Excel/CSV file into a normalized DataFrame (lowercase columns)."""
    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

    # lower columns
    df.columns = [str(c).strip().lower() for c in df.columns]

    # ensure required cols exist
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = 0

    # try to find a time column
    time_col = None
    for c in OPT_TIME_COLS:
        if c in df.columns:
            time_col = c
            break

    # keep only relevant + time if present
    keep_cols = RAW_COLS.copy()
    if time_col is not None:
        keep_cols = [time_col] + keep_cols
    df = df[keep_cols]

    # numeric cast
    for col in RAW_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if time_col is not None:
        # try parse datetime
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        except Exception:
            pass

    return df

def walk_root(root: Path):
    """Yield (server_folder, player_file_path) for all xlsx/csv under root/*/"""
    for server_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            for file in server_dir.glob(ext):
                yield server_dir.name, file

def winsorize_series(s: pd.Series, p: float) -> pd.Series:
    if len(s) == 0:
        return s
    cap = np.nanpercentile(s.to_numpy(), p)
    if np.isfinite(cap) and cap > 0:
        return np.minimum(s, cap)
    return s

def make_features_with_afk(df_raw: pd.DataFrame, winsor_p=99.5, use_logit_afk=False,
                           include_pvp=True, verbose=False) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler, list]:
    """
    Input df_raw must contain RAW_COLS (+ optional timestamp). Returns:
    - feat_unscaled: DataFrame of engineered features (log/winsor applied where applicable, BEFORE scaling)
    - feat_scaled: DataFrame of scaled features (after RobustScaler)
    - scaler: fitted RobustScaler
    - feat_cols: list of feature column names (in same order as feat_scaled columns)
    """
    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]

    # ===== 動態時間窗：AFK > 30 分鐘 → 視為 60 分鐘窗；否則 30 分鐘窗 =====
    # 原本寫死 30 分鐘與上限 1800 秒，這裡改成逐列決定 window_sec
    afk_sec = pd.to_numeric(df["afktime"], errors="coerce").fillna(0).clip(lower=0)
    window_sec = np.where(afk_sec > 1800.0, 3600.0, 1800.0)  # >30m → 60m，否則 30m
    # AFK 不能超過該筆樣本的時間窗
    afk_sec = np.minimum(afk_sec, window_sec)

    # 有效遊玩時間（分鐘）
    am = (window_sec - afk_sec) / 60.0
    am = np.clip(am, a_min=0, a_max=None)

    # AFK 比例（相對於該筆樣本實際時間窗）
    afk_ratio = (afk_sec / window_sec).clip(0, 1) 

    
    # Helper: per-active-minute rate
    def rate(col: str) -> pd.Series:
        v = df[col].fillna(0).clip(lower=0).astype(float)
        r = pd.Series(np.zeros_like(v, dtype=float), index=v.index)
        mask = am > 0
        r.loc[mask] = v.loc[mask] / am.loc[mask]
        # winsorize before log1p
        r = winsorize_series(r, winsor_p)
        # log1p
        r = np.log1p(np.clip(r, a_min=0, a_max=None))
        return r

    base_cols = [
        "pickup","block_break","tnt_prime","multi_place","chat","block_damage","block_place",
        "craft","dmg_by_entity","death","explosion","furnace_extract","inv_close","inv_open",
        "bucket_empty","bucket_fill","item_drop","exp_change","interact","level_change",
        "respawn","teleport","chunkload","redstone"
    ]
    if include_pvp and "player_death" in df.columns:
        base_cols += ["player_death"]

    feat = pd.DataFrame(index=df.index)
    for c in base_cols:
        feat[f"rate_{c}"] = rate(c)

    # Derived features built on (unlogged) rates -> but we already logged base rates.
    # For interpretability, derive from pre-log rates then log1p; so recompute pre-log rates quickly.
    def raw_rate(col: str) -> pd.Series:
        v = df[col].fillna(0).clip(lower=0).astype(float)
        r = pd.Series(np.zeros_like(v, dtype=float), index=v.index)
        mask = am > 0
        r.loc[mask] = v.loc[mask] / am.loc[mask]
        return r

    eps = 1e-6
    brk = raw_rate("block_break")
    plc = raw_rate("block_place")
    tnt = raw_rate("tnt_prime")
    exp = raw_rate("explosion")
    rds = raw_rate("redstone")
    chk = raw_rate("chunkload")
    tpl = raw_rate("teleport")
    itx = raw_rate("interact")
    bkt = raw_rate("bucket_fill") + raw_rate("bucket_empty")
    dmg = raw_rate("dmg_by_entity")
    exg = raw_rate("exp_change")
    lvl = raw_rate("level_change")
    rsp = raw_rate("respawn")
    frn = raw_rate("furnace_extract")
    cht = raw_rate("chat")
    idr = raw_rate("item_drop")
    iop = raw_rate("inv_open")
    icl = raw_rate("inv_close")
    if include_pvp and "player_death" in df.columns:
        pld = raw_rate("player_death")
    else:
        pld = pd.Series(0.0, index=df.index)

    # Domain features (then log1p)
    build_bias = np.log((plc+eps)/(brk+eps))
    explosive_intensity = np.log1p(tnt + exp)
    redstone_intensity = np.log1p(rds)
    explore_intensity  = np.log1p(chk + 0.5*tpl + 0.2*itx + 0.2*bkt)
    survival_intensity = np.log1p(dmg + exg + lvl + rsp + 0.2*frn)
    pvp_intensity      = np.log1p(pld + 0.7*dmg)
    social_intensity   = np.log1p(cht + idr + iop + icl)

    feat["build_bias"] = build_bias
    feat["explosive_intensity"] = explosive_intensity
    feat["redstone_intensity"] = redstone_intensity
    feat["explore_intensity"] = explore_intensity
    feat["survival_intensity"] = survival_intensity
    if include_pvp:
        feat["pvp_intensity"] = pvp_intensity
    feat["social_intensity"] = social_intensity

    # AFK feature
    if use_logit_afk:
        feat["afk_ratio"] = np.log((afk_ratio+1e-6)/(1-afk_ratio+1e-6))
    else:
        feat["afk_ratio"] = afk_ratio

    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(feat.values.astype("float32"))
    feat_scaled = pd.DataFrame(X_scaled, columns=feat.columns, index=feat.index)

    return feat, feat_scaled, scaler, feat.columns.tolist()

def cluster_and_profiles(feat_unscaled: pd.DataFrame, feat_scaled: pd.DataFrame, k: int, random_state: int=42):
    """Train KMeans on scaled features, return assignments, centers, silhouette, and unscaled profiles."""
    km = KMeans(n_clusters=k, n_init=50, max_iter=500, tol=1e-4, random_state=random_state)
    labels = km.fit_predict(feat_scaled.values)
    sil = np.nan
    try:
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(feat_scaled.values, labels)
    except Exception:
        pass

    centers_scaled = pd.DataFrame(km.cluster_centers_, columns=feat_scaled.columns)
    # profiles on unscaled features (mean by cluster)
    df = feat_unscaled.copy()
    df["_cluster"] = labels
    profile_unscaled = df.groupby("_cluster").mean(numeric_only=True).sort_index()

    return labels, centers_scaled, profile_unscaled, sil

def save_excel_features(out_dir: Path, feat_unscaled: pd.DataFrame, feat_scaled: pd.DataFrame, meta: dict):
    out_xlsx = out_dir / "kmeans_features.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        feat_unscaled.to_excel(w, sheet_name="features_unscaled", index=True)
        feat_scaled.to_excel(w, sheet_name="features_scaled", index=True)
        pd.DataFrame({"key": list(meta.keys()), "value": [json.dumps(meta[k], ensure_ascii=False) if isinstance(meta[k], (list, dict)) else meta[k] for k in meta]}).to_excel(w, sheet_name="meta", index=False)
    return out_xlsx

def save_excel_kmeans(out_dir: Path, assignments: pd.DataFrame, centers_scaled: pd.DataFrame,
                      profile_unscaled: pd.DataFrame, metrics: dict):
    out_xlsx = out_dir / "kmeans_results.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        assignments.to_excel(w, sheet_name="assignments", index=False)
        centers_scaled.to_excel(w, sheet_name="centers_scaled", index=False)
        profile_unscaled.to_excel(w, sheet_name="cluster_profile_unscaled", index=True)
        pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())}).to_excel(w, sheet_name="metrics", index=False)
    return out_xlsx

def main():
    ap = argparse.ArgumentParser(description="AFK-aware K-Means Excel Preprocessing")
    ap.add_argument("--root", required=True, help="根資料夾（底下為 server 資料夾）")
    ap.add_argument("--out", required=True, help="輸出資料夾")
    ap.add_argument("--winsor_p", type=float, default=99.5, help="winsor 百分位（0~100）")
    ap.add_argument("--include_pvp", action="store_true", help="是否納入 PVP 代理特徵（player_death）")
    ap.add_argument("--use_logit_afk", action="store_true", help="AFK 使用 logit 轉換（預設使用比例）")
    ap.add_argument("--train_kmeans", action="store_true", help="是否同時訓練 KMeans")
    ap.add_argument("--k", type=int, default=9, help="K 值（train_kmeans 時有效）")
    ap.add_argument("--random_state", type=int, default=42, help="隨機種子")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists() or not root.is_dir():
        print(f"[ERROR] 根資料夾不存在或不是資料夾：{root}", file=sys.stderr)
        sys.exit(2)

    rows = []
    meta_rows = []
    for server_name, file in walk_root(root):
        player_id = file.stem
        try:
            df = read_one_table(file)
        except Exception as e:
            warnings.warn(f"讀取失敗略過：{file} -> {e}")
            continue
        df.insert(0, "server", server_name)
        df.insert(1, "player_id", player_id)

        # add a row id if no time col
        has_time = any(c in df.columns for c in OPT_TIME_COLS)
        if not has_time:
            df.insert(2, "row_idx", np.arange(len(df)))
        rows.append(df)

    if not rows:
        print("[ERROR] 沒有讀到任何檔案，請確認資料夾與副檔名（xlsx/xls/csv）。", file=sys.stderr)
        sys.exit(3)

    df_all = pd.concat(rows, axis=0, ignore_index=True)
    # save raw concat for reference
    df_all.to_parquet(out_dir / "raw_concat.parquet", index=False)

    # Keep metadata cols for join later
    meta_cols = [c for c in ["server","player_id","timestamp","time","ts","datetime","row_idx"] if c in df_all.columns]
    df_meta = df_all[meta_cols].copy()
    df_raw = df_all.drop(columns=[c for c in meta_cols if c in df_all.columns], errors="ignore")

    # Feature engineering
    feat_unscaled, feat_scaled, scaler, feat_cols = make_features_with_afk(
        df_raw,
        winsor_p=args.winsor_p,
        use_logit_afk=args.use_logit_afk,
        include_pvp=args.include_pvp
    )

    # 過濾掉完全 AFK 的樣本
    mask = feat_unscaled["afk_ratio"] < 1.0
    feat_unscaled = feat_unscaled[mask]
    feat_scaled = feat_scaled[mask]
    df_meta = df_meta.loc[mask]


    # Attach meta back for export convenience (only to unscaled; scaled keeps pure features)
    feat_unscaled_export = pd.concat([df_meta.reset_index(drop=True), feat_unscaled.reset_index(drop=True)], axis=1)

    # Save features excel + params
    meta_info = {
        "winsor_p": args.winsor_p,
        "include_pvp": bool(args.include_pvp),
        "use_logit_afk": bool(args.use_logit_afk),
        "feature_count": len(feat_cols),
        "feature_columns": feat_cols,
        "note": "features_unscaled: 已對 rate 類別做 winsor + log1p；衍生亦已 log1p；afk_ratio 依參數為比例或 logit。features_scaled: RobustScaler 標準化後的值。"
    }
    out_feat_xlsx = save_excel_features(out_dir, feat_unscaled_export, pd.DataFrame(feat_scaled, columns=feat_cols), meta_info)

    # Save scaler parameters & feature columns
    scaler_params = {
        "center_": scaler.center_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "with_centering": scaler.with_centering,
        "with_scaling": scaler.with_scaling,
        "quantile_range": (25.0, 75.0)
    }
    (out_dir / "scaler_params.json").write_text(json.dumps(scaler_params, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "columns.json").write_text(json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 特徵已輸出：{out_feat_xlsx}")

    if args.train_kmeans:
        # 1) 分群時剔除 AFK 特徵（避免 AFK 吞掉行為差異）
        feat_scaled_for_km = feat_scaled

# 2) 對探險/紅石相關特徵做「加權」（乘法放大）
        boosts = {
            "rate_chunkload": 1.0,
            "explore_intensity": 1.0,
            "rate_teleport": 1.0,
            "rate_redstone": 1.5,
            "redstone_intensity": 1.3,
            # 你也可視狀況加： "rate_interact": 1.15
        }
        for col, w in boosts.items():
            if col in feat_scaled_for_km.columns:
                feat_scaled_for_km.loc[:, col] *= w

# 3) 用調整後的特徵做 K-Means
        labels, centers_scaled, profile_unscaled, sil = cluster_and_profiles(
            feat_unscaled=feat_unscaled,
            feat_scaled=feat_scaled_for_km,  # ← 改用這個
            k=args.k,
            random_state=args.random_state
        )


        # Build assignments table with meta
        assign = df_meta.copy()
        assign["_cluster"] = labels
        # Save KMeans results
        metrics = {"silhouette": float(sil), "k": int(args.k)}
        out_km_xlsx = save_excel_kmeans(out_dir, assignments=assign, centers_scaled=centers_scaled,
                                        profile_unscaled=profile_unscaled, metrics=metrics)
        print(f"[OK] KMeans 已輸出：{out_km_xlsx}")
        print(f"[METRICS] silhouette={sil:.4f}  K={args.k}")

if __name__ == "__main__":
    main()


