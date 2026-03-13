# 2026TISF_Kmeans

以 **K-Means** 為核心的玩家行為分群與標註工具組。專案流程：將原始玩家事件資料經過特徵工程 → 分群 → 產出帶有玩家類型標籤的結果，並輸出 Excel 以銜接迴歸/預測模型。

---

## 功能

- **可重複的 K-Means 分群**：針對多玩家 CSV進行特徵聚合與分群。  
- **特徵工程模組**：數據清理、標準化、Winsorization 等步驟。（對應資料夾：`特徵工程/`）
- **特徵標註 & 匯出**：將分群結果映射到對應的玩家類型，並一鍵輸出 Excel（可自訂輸出資料夾`export_to_excel.py`）。

---

## 專案結構

```
2026TISF_Kmeans/
├─ comp-forecast-service/     
├─ 特徵工程/                  
├─ kmeans test result/        
├─ excel/                     
├─ backup sql/                 
├─ export_to_excel.py          
├─ labeled_results.xlsx        
├─ README.md                  
└─ LICENSE                     
```

---

## 安裝需求

- Python 3.10+（建議）  
- 主要套件：  
  - `pandas`, `numpy`  
  - `scikit-learn`（KMeans、標準化...等）  
  - `openpyxl`（Excel 匯出）或 `xlsxwriter`  
  - `matplotlib` / `plotly`

安裝範例：
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pandas numpy scikit-learn openpyxl
pip install matplotlib
```

---

## 建議資料格式

- **輸入**：每位玩家一份 CSV 或同一張資料表中含 PlayerID 欄，時間窗為 30–60 分鐘或 1 小時記錄一次。  
- **常見欄位**：  
  - `player_id, ts_window_start, blocks_placed, blocks_broken, chunk_loads, tnt_exploded, entity_kills, items_picked, items_dropped, container_interactions, chat_count, afk_minutes, active_minutes, ...`  
- **前處理**：  
  - 缺值處理 / 去極值 / 標準化（`StandardScaler`）  
  - 對數轉換 / RobustScaler

---

## 專案流程

1. **特徵工程**  
   - 在 `特徵工程/` 內呼叫已有的特徵預處理。  
2. **分群（K-Means）**  
   - 指定 `k`群，設定 `random_state` 與 `n_init`。  
3. **類型對映（標註）**  
   - 將 ClusterID 對映到玩家類型（建築/探險/生存/紅石/競技/破壞者/社交/掛機）。  
4. **輸出成果**  
   - 以 `export_to_excel.py` 將結果輸出到 `excel/` 與 `labeled_results.xlsx`。

---

## 快速開始

```bash
# 1) 準備資料
mkdir -p data excel "kmeans test result"

# 2) 執行分群
python -m 特徵工程.run_kmeans --input data --k 8 --random_state 42

# 3) 映射玩家類型
python -m 特徵工程.label_clusters --input clustered.parquet --mapping config/cluster_to_playertype.yaml

# 4) 匯出 Excel
python export_to_excel.py --input labeled.parquet --output labeled_results.xlsx
```

---

## 輸出說明

- `labeled_results.xlsx`：每列包含玩家/時間窗與對應的 ClusterID 與玩家類型標籤，可用於
  
---

## 授權

本專案使用 **Apache-2.0** 授權，詳見 [`LICENSE`](./LICENSE)。
