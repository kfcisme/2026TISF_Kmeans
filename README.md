# 2026TISF_Kmeans

以 **K-Means** 為核心的玩家行為分群與標註工具組。專案聚焦於：將原始玩家事件資料經過特徵工程 → 分群 → 產出帶有玩家類型標籤的結果，並支援輸出 Excel 報表以銜接後續迴歸/預測模型與海報圖表產製。

> 授權條款：Apache-2.0（已在 repo 中標示）。

---

## 功能亮點

- **可重複的 K-Means 分群流程**：針對多玩家 CSV/資料表執行特徵聚合與分群，支援固定隨機種子以確保可重現性。  
- **特徵工程模組**：集中處理常見的清理、尺度化、Winsorization 等步驟，方便替換與擴充。（對應資料夾：`特徵工程/`）
- **結果標註與匯出**：將分群結果對映到自定義玩家類型（如建築/探險/社交/紅石等），並可一鍵輸出 Excel（`export_to_excel.py`）。
- **結果留存**：`kmeans test result/` 內可保存中間與最終成果，利於回溯與海報製作。

---

## 專案結構

```
2026TISF_Kmeans/
├─ comp-forecast-service/      # （預留）分群後的比較/預測服務腳本與範例
├─ 特徵工程/                   # 特徵工程相關腳本與轉換器
├─ kmeans test result/         # 分群與標註後的輸出樣本與測試結果
├─ excel/                      # 匯出/範例報表存放區
├─ backup sql/                 # （可選）SQL 匯出/備份
├─ export_to_excel.py          # 將標註結果輸出為 Excel 的工具腳本
├─ labeled_results.xlsx        # 範例：已標註的輸出檔
├─ README.md                   # 本說明文件
└─ LICENSE                     # Apache-2.0 授權
```

---

## 安裝需求

- Python 3.10+（建議）  
- 主要套件：  
  - `pandas`, `numpy`  
  - `scikit-learn`（KMeans、標準化、管線化等）  
  - `openpyxl`（Excel 匯出）或 `xlsxwriter`  
  - （選配）`matplotlib` / `plotly`（視覺化）

安裝範例：
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pandas numpy scikit-learn openpyxl
# 如需視覺化
pip install matplotlib
```

---

## 資料格式（建議）

- **輸入**：每位玩家一份 CSV 或同一張資料表中含 PlayerID 欄，時間窗為 30–60 分鐘或 1 小時聚合一次。  
- **常見欄位**：  
  - `player_id, ts_window_start, blocks_placed, blocks_broken, chunk_loads, tnt_exploded, entity_kills, items_picked, items_dropped, container_interactions, chat_count, afk_minutes, active_minutes, ...`  
- **前處理**：  
  - 缺值處理 / 去極值 / 標準化（`StandardScaler`）  
  - 可加入對數轉換、RobustScaler、分位數縮放等

---

## 典型流程

1. **特徵工程**  
   - 在 `特徵工程/` 內撰寫或呼叫既有轉換器。  
2. **分群（K-Means）**  
   - 指定 `k`（例如 6–8 群），設定 `random_state` 與 `n_init`。  
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

# 3) 對映玩家類型
python -m 特徵工程.label_clusters --input clustered.parquet --mapping config/cluster_to_playertype.yaml

# 4) 匯出 Excel
python export_to_excel.py --input labeled.parquet --output labeled_results.xlsx
```

---

## 輸出說明

- `labeled_results.xlsx`：每列包含玩家/時間窗與對應的 ClusterID 與 玩家類型標籤，可用於：  
  - 訓練 MLR/LSTM 模型建立負載關係式  
  - 生成海報圖表  
  - 模擬動態調節伺服器數量

---

## 實證與結果

- 不同 `k` 值對 SSE/輪廓係數的影響  
- 各玩家類型的特徵概況（如 chunk_loads 高→探險，blocks_placed 高→建築）  
- 與「僅用玩家人數」模型的效果比較  
- 對伺服器 TPS 維持與節能的改善情形

---

## 路線圖

- [ ] 補齊 `requirements.txt`
- [ ] 新增 demo notebook
- [ ] 加入評估指標自動報告
- [ ] 串接動態調節模擬專案
- [ ] 加入自動化參數搜尋

---

## 授權

本專案使用 **Apache-2.0** 授權，詳見 [`LICENSE`](./LICENSE)。
