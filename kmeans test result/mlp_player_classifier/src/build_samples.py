from pathlib import Path
import pandas as pd
import numpy as np

# ========== 路徑請依你的電腦調整 ==========
ROOT = Path(r"C:\Users\hsu96\OneDrive\Desktop\mysql_csv_exports")
DIR_MLP   = ROOT / "mlp_player_classifier"
DIR_TEST  = ROOT / "test"   # 有 kmeans_features.xlsx / labeled.xlsx / kmeans_results.xlsx
OUT_DIR   = DIR_MLP / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_KFEATS   = DIR_TEST / "kmeans_features.xlsx"
FILE_LABELED  = DIR_TEST / "labeled.xlsx"
FILE_KRESULTS = DIR_TEST / "kmeans_results.xlsx"  # 可選

def read_xlsx(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_excel(path)

def has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)

def main():
    # 1) 讀資料
    dfX = read_xlsx(FILE_KFEATS)    # 特徵：server, player_id, row_idx, rate_*
    dfY = read_xlsx(FILE_LABELED)   # 標籤：server, player_id, row_idx, final_label(或label), dist_*...

    # 2) 決定 join key
    prefer_keys = ["server","player_id","row_idx"]
    time_keys_x = ["player_id","time"]
    time_keys_y = ["player_id","time"]

    if has_cols(dfX, prefer_keys) and has_cols(dfY, prefer_keys):
        join_keys = prefer_keys
    elif has_cols(dfX, time_keys_x) and has_cols(dfY, time_keys_y):
        join_keys = time_keys_x  # (= time_keys_y)
    elif has_cols(dfX, ["player_id"]) and has_cols(dfY, ["player_id"]):
        join_keys = ["player_id"]
    else:
        raise RuntimeError(f"找不到可用的合併鍵，請檢查欄位。X 有 {list(dfX.columns)[:10]}..., Y 有 {list(dfY.columns)[:10]}...")

    # 3) 選標籤欄位（final_label > label）
    label_col = "final_label" if "final_label" in dfY.columns else ("label" if "label" in dfY.columns else None)
    if label_col is None:
        raise RuntimeError("labeled.xlsx 中找不到 final_label 或 label 欄位")

    # 4) 合併 X 與 Y（只帶上需要的 Y 欄）
    keep_y = list(dict.fromkeys(join_keys + [label_col]))
    df = pd.merge(dfX, dfY[keep_y].drop_duplicates(), on=join_keys, how="inner")

    # 5) 可選：把 kmeans_results 也併進來（例如 _cluster）
    if FILE_KRESULTS.exists():
        dfK = read_xlsx(FILE_KRESULTS)
        if has_cols(dfK, join_keys):
            keep_k = list(dict.fromkeys(join_keys + [c for c in dfK.columns if c.startswith(("_cluster","kmeans","dist_","km_"))]))
            if len(keep_k) > len(join_keys):
                df = pd.merge(df, dfK[keep_k].drop_duplicates(), on=join_keys, how="left")

    # 6) AFK 衍生（若有 afktime/或 rate_afktime）
    if "afktime" in df.columns and "AFK_ratio" not in df.columns:
        df["AFK_ratio"] = df["afktime"] / 1800.0
    if "afktime" in df.columns and "active_minutes" not in df.columns:
        df["active_minutes"] = np.maximum(0.0, 30.0 - df["afktime"]/60.0)

    # 7) 準備輸出欄（player_id + 主鍵 + 標籤 + 特徵）
    #    特徵預設：所有以 rate_ 開頭者 + 可選的 dist_* / _cluster / pmax
    feature_cols = [c for c in df.columns if c.startswith("rate_")]
    feature_cols += [c for c in df.columns if c.startswith("dist_")]
    for extra in ["_cluster","pmax","dist_to_assigned","min_dist","AFK_ratio","active_minutes"]:
        if extra in df.columns and extra not in feature_cols:
            feature_cols.append(extra)

    # 確保 player_id 在
    id_cols = [c for c in ["server","player_id","row_idx","time"] if c in join_keys or c in df.columns]
    base_cols = list(dict.fromkeys(["player_id"] + id_cols))
    out_cols = list(dict.fromkeys(base_cols + [label_col] + feature_cols))

    df_out = df[out_cols].copy()
    df_out.rename(columns={label_col: "label"}, inplace=True)

    out_path = OUT_DIR / "samples.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("✅ 產生完成：", out_path)
    print(f"   共有 {len(df_out)} 列、{len(df_out.columns)} 欄")
    print("   join keys =", join_keys)
    print("   label 欄位 =", label_col)
    print("   特徵欄數量 =", len(feature_cols))

if __name__ == "__main__":
    main()
