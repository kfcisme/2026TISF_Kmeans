

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_sheet_tolerant(xlsx_path: Path, candidates: list[str]) -> pd.DataFrame:
    last_err = None
    for name in candidates:
        try:
            return pd.read_excel(xlsx_path, sheet_name=name)
        except Exception as e:
            last_err = e
    xls = pd.ExcelFile(xlsx_path)
    raise RuntimeError(
        f"cant find {xlsx_path.name}  sheet（try {candidates} useable sheets: {xls.sheet_names}\n error: {last_err}"
    )

def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if cols:
        print(f"[INFO] delete coloums{cols}")
        df = df.drop(columns=cols)
    return df

def maybe_apply_columns_json(df: pd.DataFrame, columns_json: Path) -> pd.DataFrame:
    if not columns_json.exists():
        return df

    try:
        target_cols = json.loads(columns_json.read_text(encoding="utf-8"))
        if not isinstance(target_cols, list):
            print("[WARN] columns.json isnt list, skip")
            return df
    except Exception as e:
        print(f"[WARN] unable to load columns.json. Reason: {e}")
        return df

    cur_n = df.shape[1]
    tgt_n = len(target_cols)

    if cur_n == tgt_n:
        df.columns = target_cols
        print(f"[OK] Finish")
        return df

    # 特例
    if cur_n == tgt_n + 1:
        first_col = df.columns[0]
        first_series = df.iloc[:, 0]
        looks_like_index = (
            str(first_col).lower() in {"index", "", "id"} or
            pd.api.types.is_integer_dtype(first_series) and
            (first_series.reset_index(drop=True).diff().dropna() == 1).all()
        )
        if looks_like_index:
            df = df.iloc[:, 1:].copy()
            if df.shape[1] == tgt_n:
                df.columns = target_cols
                print(f"[OK] finish")
                return df

    print(f"[WARN]")
    return df

def align_center_to_features(features: pd.DataFrame, centers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fcols = list(features.columns)
    ccols = list(centers.columns)

    common = [c for c in fcols if c in ccols]
    missing_in_centers = [c for c in fcols if c not in ccols]
    extra_in_centers = [c for c in ccols if c not in fcols]

    if missing_in_centers:
        print(f"[WARN] centers missing feature, cover 0 instead{missing_in_centers}")


    centers_aligned = centers.reindex(columns=fcols, fill_value=0.0)
    features_aligned = features[fcols]  # 保持 features 原欄序
    return features_aligned, centers_aligned


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(abs(w) for w in weights.values()) or 1.0
    return {k: w / total for k, w in weights.items()}

def default_axes(feature_names: list[str]) -> dict[str, dict[str, float]]:
    has = set(feature_names).__contains__
    axes = {
        "Build": {
            "rate_block_place": 1.0,
            **({"rate_multi_place": 0.8} if has("rate_multi_place") else {}),
            **({"rate_craft": 0.8} if has("rate_craft") else {}),
            **({"rate_furnace_extract": 0.6} if has("rate_furnace_extract") else {}),
            **({"build_bias": 0.8} if has("build_bias") else {}),
            **({"rate_block_break": -0.2} if has("rate_block_break") else {}),
        },
        "Exploration": {
            **({"rate_chunkload": 1.0} if has("rate_chunkload") else {}),
            **({"rate_teleport": 0.6} if has("rate_teleport") else {}),
            **({"rate_interact": 0.3} if has("rate_interact") else {}),
            **({"rate_bucket_fill": 0.2} if has("rate_bucket_fill") else {}),
            **({"rate_bucket_empty": 0.2} if has("rate_bucket_empty") else {}),
            **({"explore_intensity": 0.6} if has("explore_intensity") else {}),
        },
        "Survival": {
            **({"rate_dmg_by_entity": 0.8} if has("rate_dmg_by_entity") else {}),
            **({"rate_exp_change": 0.6} if has("rate_exp_change") else {}),
            **({"rate_level_change": 0.6} if has("rate_level_change") else {}),
            **({"rate_respawn": 0.6} if has("rate_respawn") else {}),
            **({"rate_furnace_extract": 0.2} if has("rate_furnace_extract") else {}),
            **({"survival_intensity": 0.8} if has("survival_intensity") else {}),
        },
        "Redstone": {
            **({"rate_redstone": 1.0} if has("rate_redstone") else {}),
            **({"rate_craft": 0.8} if has("rate_craft") else {}),
            **({"redstone_intensity": 0.6} if has("redstone_intensity") else {}),
        },
        "PvP": {
            **({"rate_player_death": 0.7} if has("rate_player_death") else {}),
            **({"rate_dmg_by_entity": 0.6} if has("rate_dmg_by_entity") else {}),
            **({"pvp_intensity": 0.8} if has("pvp_intensity") else {}),
        },
        "Explosive": {
            **({"rate_tnt_prime": 0.9} if has("rate_tnt_prime") else {}),
            **({"rate_explosion": 0.9} if has("rate_explosion") else {}),
            **({"rate_block_damage": 0.3} if has("rate_block_damage") else {}),
            **({"explosive_intensity": 0.6} if has("explosive_intensity") else {}),
        },
        "Social": {
            **({"rate_chat": 0.8} if has("rate_chat") else {}),
            **({"rate_item_drop": 0.4} if has("rate_item_drop") else {}),
            **({"rate_inv_open": 0.4} if has("rate_inv_open") else {}),
            **({"rate_inv_close": 0.4} if has("rate_inv_close") else {}),
            **({"social_intensity": 0.6} if has("social_intensity") else {}),
        },
        "AFK": {
            **({"afk_ratio": 1.0} if has("afk_ratio") else {}),
        },
    }
    # delete
    return {ax: w for ax, w in axes.items() if w}



def load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config YAML 內容需為 key-value 結構")
    return cfg

def main():

    '''fix by chatgpt'''
    ap = argparse.ArgumentParser(description="K-Means 後設語意標籤化（強化防呆版）")
    ap.add_argument("--out", required=True, help="前一個 pipeline 的輸出資料夾（內含 kmeans_features.xlsx / kmeans_results.xlsx）")
    ap.add_argument("--config", help="語意軸與門檻的 YAML 設定路徑", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    if not out_dir.exists():
        raise FileNotFoundError(f"cant find output folders {out_dir}")

    cfg = load_config(Path(args.config)) if args.config else {}
    unknown_percentile = float(cfg.get("unknown_percentile", 95.0))
    softmax_confidence_min = float(cfg.get("softmax_confidence_min", 0.5))
    secondary_gap_threshold = float(cfg.get("secondary_gap_threshold", 0.1))
    tau_cfg = cfg.get("tau", None)
    top_m = int(cfg.get("top_m", 5))

    feat_xlsx = out_dir / "kmeans_features.xlsx"
    res_xlsx  = out_dir / "kmeans_results.xlsx"
    if not feat_xlsx.exists() or not res_xlsx.exists():
        raise FileNotFoundError(f"cant find files {feat_xlsx if not feat_xlsx.exists() else ''} {res_xlsx if not res_xlsx.exists() else ''}")

    features_scaled = read_sheet_tolerant(feat_xlsx, ["features_scaled", "scaled", "feat_scaled"])
    centers_scaled  = read_sheet_tolerant(res_xlsx,  ["centers_scaled", "centers", "cluster_centers"])
    assignments     = read_sheet_tolerant(res_xlsx,  ["assignments", "labels", "clusters"])

    # clear Unnamed
    features_scaled = drop_unnamed(features_scaled)
    centers_scaled  = drop_unnamed(centers_scaled)

    cols_json = out_dir / "columns.json"
    features_scaled = maybe_apply_columns_json(features_scaled, cols_json)

    # 對齊
    features_scaled, centers_scaled = align_center_to_features(features_scaled, centers_scaled)

    cluster_col = None
    for cand in ["_cluster", "cluster", "label", "labels", "cluster_id"]:
        if cand in assignments.columns:
            cluster_col = cand
            break
    if cluster_col is None:
        raise RuntimeError(f"cant find assignments missing: _cluster / cluster / label, {list(assignments.columns)}")


    if centers_scaled.index.name is not None:
        centers_scaled = centers_scaled.reset_index(drop=True)
    centers_scaled.index = range(len(centers_scaled))

    feat_cols = list(features_scaled.columns)
    axes = cfg.get("axes") or default_axes(feat_cols)

    def norm_w(wdict):
        s = sum(abs(v) for v in wdict.values()) or 1.0
        return {k: v / s for k, v in wdict.items()}

    A = centers_scaled.reindex(columns=feat_cols, fill_value=0.0).to_numpy(dtype=float)
    axis_scores = {}
    for ax, w in axes.items():
        vec = np.zeros(len(feat_cols), dtype=float)
        for f, v in norm_w(w).items():
            if f in feat_cols:
                vec[feat_cols.index(f)] = v
        axis_scores[ax] = A @ vec
    axis_scores_df = pd.DataFrame(axis_scores, index=centers_scaled.index)

    # 主/副標籤
    primary = axis_scores_df.idxmax(axis=1)
    secondary = axis_scores_df.apply(lambda r: r.sort_values(ascending=False).index[1] if r.size >= 2 else r.index[0], axis=1)
    delta = axis_scores_df.max(axis=1) - axis_scores_df.apply(lambda r: r.sort_values(ascending=False).iloc[1], axis=1)

    allow_composite = bool(cfg.get("allow_composite", True))
    labels = [p if (allow_composite is False or d >= secondary_gap_threshold) else f"{p} / {s}"
          for p, s, d in zip(primary, secondary, delta)]


    cluster_labels = pd.DataFrame({
        "_cluster": centers_scaled.index,
        "primary_label": primary.values,
        "secondary_label": secondary.values,
        "label": labels
    }).join(axis_scores_df, how="left")

    X = features_scaled.reindex(columns=feat_cols, fill_value=0.0).to_numpy(dtype=float)
    C = centers_scaled.reindex(columns=feat_cols, fill_value=0.0).to_numpy(dtype=float)
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    dmin = np.sqrt(d2.min(axis=1))

    assigned = assignments[cluster_col].to_numpy()
    try:
        assigned = assigned.astype(int)
    except Exception:
        assigned = pd.to_numeric(assigned, errors="coerce").fillna(-1).astype(int)

    valid_mask = (assigned >= 0) & (assigned < C.shape[0])
    if not np.all(valid_mask):
        bad = np.where(~valid_mask)[0][:10]
        print(f"[WARN] assignments error")
        assigned[~valid_mask] = d2.argmin(axis=1)[~valid_mask]

    d_assigned = np.sqrt(d2[np.arange(d2.shape[0]), assigned])

    # 軟指派
    tau = float(tau_cfg) if tau_cfg is not None else float(np.median(dmin) + 1e-8)
    logits = -d2 / (tau ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
    pmax = P.max(axis=1)

    unknown_thr = float(np.percentile(dmin, unknown_percentile))
    unknown = (dmin > unknown_thr) | (pmax < softmax_confidence_min)

    # conbine and output
    label_map = dict(zip(cluster_labels["_cluster"], cluster_labels["label"]))
    out_assign = assignments.copy()
    out_assign["label"] = [label_map.get(int(c), f"Cluster-{int(c)}") for c in assigned]
    out_assign["dist_to_assigned"] = d_assigned
    out_assign["min_dist"] = dmin
    out_assign["pmax"] = pmax
    out_assign["is_unknown"] = unknown
    cfg_override = cfg.get("sample_override", {})
    features_scaled_df = features_scaled.copy()

    def apply_sample_override(label_series: pd.Series, feats: pd.DataFrame, spec: dict, new_label: str) -> pd.Series:

        cond = pd.Series(True, index=feats.index)

        byp = spec.get("by_percentile", {})
        for f, pct in byp.items():
            if f in feats.columns:
                thr = feats[f].quantile(float(pct) / 100.0)
                cond &= feats[f] >= thr
            else:
                cond &= False
        any_feats = spec.get("also_require_any", [])
        if any_feats:
            any_ok = pd.Series(False, index=feats.index)
            for f in any_feats:
                if f in feats.columns:
                    any_ok |= feats[f] >= feats[f].median()
            cond &= any_ok

        target_mask = cond & (~out_assign["is_unknown"])
        label_series.loc[target_mask] = new_label
        return label_series

    def prettify(lbl: str) -> str:
        return "PvP" if lbl.lower() == "pvp" else lbl.capitalize()

    for key, spec in cfg_override.items():
        out_assign["label"] = apply_sample_override(out_assign["label"], features_scaled_df, spec, prettify(key))


    legend_rows = []
    for axis, wmap in axes.items():
        for fname, w in normalize_weights(wmap).items():
            legend_rows.append({"axis": axis, "feature": fname, "weight": w})
    legend_df = pd.DataFrame(legend_rows)

    # save
    out_labeled = out_dir / "labeled_results.xlsx"
    with pd.ExcelWriter(out_labeled, engine="openpyxl") as w:
        out_assign.to_excel(w, sheet_name="assignments_labeled", index=False)
        cluster_labels.to_excel(w, sheet_name="cluster_labels", index=False)
        axis_scores_df.to_excel(w, sheet_name="axis_scores", index=True)
        legend_df.to_excel(w, sheet_name="axis_legend", index=False)

    print(f"[OK] 已輸出：{out_labeled}")
    print(f"[INFO] Unknown 距離閾值（{unknown_percentile}% 分位）= {unknown_thr:.6f}，tau = {tau:.6f}")

if __name__ == "__main__":
    main()
