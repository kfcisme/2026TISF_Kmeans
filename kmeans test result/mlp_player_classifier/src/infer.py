import yaml, pandas as pd, numpy as np
import onnxruntime as ort
from features import add_derived_features, winsorize, log1p_cols, robust_scale

def build_preprocess(df, C, feat_cols):
    df = add_derived_features(df)
    if C["winsor"]["enabled"]:
        df = winsorize(df, feat_cols, C["winsor"]["p_low"], C["winsor"]["p_high"])
    if C["transform"]["log1p"]:
        df = log1p_cols(df, feat_cols)
    if C["transform"]["robust_scale"]:
        df = robust_scale(df, feat_cols)
    return df

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def main():
    C = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
    # 載入 ckpt 資訊
    import torch
    ckpt = torch.load("best.pt", map_location="cpu")
    feat_cols = ckpt["feat_cols"]
    classes = ckpt["classes"]

    # 載入 ONNX
    sess = ort.InferenceSession(C["export"]["onnx_path"], providers=["CPUExecutionProvider"])

    # 假設有一個新時段的資料 new.csv（同 schema）
    df_new = pd.read_csv("data/new.csv")
    df_new = build_preprocess(df_new, C, feat_cols)
    X = df_new[feat_cols].to_numpy(np.float32)
    logits = sess.run(["logits"], {"features": X})[0]
    prob = softmax(logits)
    pred_idx = prob.argmax(1)
    df_new["pred_label"] = [classes[i] for i in pred_idx]
    df_new["pred_conf"] = prob.max(1)

    # 可選：低信心標 Unknown
    df_new["final_label"] = np.where(df_new["pred_conf"]>=0.55, df_new["pred_label"], "Unknown")
    df_new.to_csv("data/new_with_preds.csv", index=False)
    print("Saved to data/new_with_preds.csv")

if __name__ == "__main__":
    main()
