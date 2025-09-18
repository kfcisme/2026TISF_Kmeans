# -*- coding: utf-8 -*-
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import os
import numpy as np

# 可選：ONNX runtime / Torch（若你之後要換 LSTM/ONNX）
USE_ONNX = os.getenv("USE_ONNX", "false").lower() == "true"
ONNX_PATH = os.getenv("ONNX_PATH", "lstm_comp.onnx")
USE_TORCH = os.getenv("USE_TORCH", "false").lower() == "true"

sess = None
if USE_ONNX:
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        print(f"[LOAD] ONNX session loaded: {ONNX_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load ONNX: {e}; fallback to baseline")
        sess = None
elif USE_TORCH:
    try:
        import torch
        from model_lstm import SimpleLSTM
        # 這裡示範讀一個 state_dict；實務請換成你的檔名
        CKPT = os.getenv("TORCH_CKPT", "lstm_comp.pt")
        D = int(os.getenv("INPUT_DIM", "9"))  # 8 類比例 + 1 總上線N
        H = int(os.getenv("HIDDEN", "64"))
        CLASSES = 8
        model = SimpleLSTM(D, H, CLASSES)
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        model.eval()
        print(f"[LOAD] Torch LSTM loaded: {CKPT}")
    except Exception as e:
        print(f"[WARN] Failed to load Torch model: {e}; fallback to baseline")
        USE_TORCH = False
        model = None

app = FastAPI(title="Composition Forecast Service", version="1.0.0")

class Req(BaseModel):
    server_id: str = Field(..., description="伺服器ID")
    comp_seq: List[List[float]] = Field(..., description="最近L個視窗的8維比例，每列和=1")
    n_seq: Optional[List[int]] = Field(None, description="最近L個視窗的總上線人數（可選）")
    horizon: int = Field(1, ge=1, le=6, description="往前預測幾個視窗，預設1")

    @validator("comp_seq")
    def check_comp(cls, v):
        if len(v) < 2:
            raise ValueError("comp_seq 至少要2個時間點（建議≥6）")
        for row in v:
            if len(row) != 8:
                raise ValueError("每個 comp 向量長度必須為8（AFK..Survival）")
            s = sum(row)
            if s <= 0:
                raise ValueError("comp 向量不可全0")
        return v

class Resp(BaseModel):
    p_hat: List[List[float]]  # shape: (horizon, 8)
    method: str               # "baseline" / "onnx" / "torch"

def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 1e-12)
    x = x / x.sum(axis=-1, keepdims=True)
    return x

def baseline_predict(comp_seq: np.ndarray, horizon: int) -> np.ndarray:
    """
    comp_seq: (L, 8)
    簡單 baseline：後3個視窗均值作為 t+1；遞迴產生多步。
    """
    seq = comp_seq.copy()
    outs = []
    for _ in range(horizon):
        if len(seq) >= 3:
            pred = seq[-3:].mean(axis=0)
        else:
            pred = seq[-1]
        pred = pred / pred.sum()
        outs.append(pred)
        seq = np.concatenate([seq, pred[None, :]], axis=0)
    return np.vstack(outs)

@app.get("/health")
def health():
    backend = "baseline"
    if sess is not None: backend = "onnx"
    elif USE_TORCH: backend = "torch"
    return {"ok": True, "backend": backend}

@app.post("/forecast_next_comp", response_model=Resp)
def forecast_next_comp(req: Req):
    comp_seq = np.array(req.comp_seq, dtype=np.float32)  # (L, 8)
    horizon = int(req.horizon)

    # 後端選擇：ONNX > TORCH > baseline
    method = "baseline"
    if sess is not None:
        try:
            import onnxruntime as ort
            # 假設 ONNX 輸入是 (1, L, 9)；第9維是 N（若沒有N，就填總上線平均或0）
            if req.n_seq is not None:
                n = np.array(req.n_seq, dtype=np.float32).reshape(-1, 1)
            else:
                # 沒有N就用全1（或歷史平均），避免模型崩潰
                n = np.ones((comp_seq.shape[0], 1), dtype=np.float32)
            x = np.concatenate([comp_seq, n], axis=1)[None, ...]  # (1, L, 9)
            out = sess.run(None, {"input": x})
            p1 = out[0]  # 期望輸出為 (1, 8) 或 (1, H, 8)
            if p1.ndim == 3: p1 = p1[:, -1, :]
            pred = _normalize_simplex(p1[0])
            # 多步遞迴
            preds = [pred]
            seq_comp = comp_seq.copy()
            for _ in range(horizon - 1):
                seq_comp = np.concatenate([seq_comp, pred[None, :]], axis=0)
                # 之後每步都用最近 L 步丟進 ONNX（為簡化此處用 baseline 遞迴，也可再跑ONNX）
                pred = baseline_predict(seq_comp, 1)[0]
                preds.append(pred)
            method = "onnx"
            return Resp(p_hat=[p.tolist() for p in preds], method=method)
        except Exception as e:
            print(f"[WARN] ONNX run failed: {e}. Fallback baseline.")

    if USE_TORCH:
        try:
            import torch
            from model_lstm import prepare_input
            x = prepare_input(comp_seq, req.n_seq)  # (1, L, 9)
            with torch.no_grad():
                logits = model(x)      # (1, 8)
                p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            preds = [p]
            seq_comp = comp_seq.copy()
            for _ in range(horizon - 1):
                seq_comp = np.concatenate([seq_comp, p[None, :]], axis=0)
                p = baseline_predict(seq_comp, 1)[0]
                preds.append(p)
            method = "torch"
            return Resp(p_hat=[p.tolist() for p in preds], method=method)
        except Exception as e:
            print(f"[WARN] Torch run failed: {e}. Fallback baseline.")

    # Baseline
    preds = baseline_predict(comp_seq, horizon)
    return Resp(p_hat=[p.tolist() for p in preds], method=method)
