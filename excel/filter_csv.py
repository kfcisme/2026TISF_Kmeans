import pandas as pd
import os


INPUT_DIR = "C:\\Users\\hsu96\\OneDrive\\Desktop\\mysql_csv_exports\\excel\\logplayerplugin_prod"         
OUTPUT_DIR = "C:\\Users\\hsu96\\OneDrive\\Desktop\\mysql_csv_exports\\excel\\filtered\\logplayerplugin_prod"    

os.makedirs(OUTPUT_DIR, exist_ok=True)


exclude_cols = ["record_time", "cmd_pre", "cmd_send"]


for file in os.listdir(INPUT_DIR):
    if file.endswith(".csv"):
        file_path = os.path.join(INPUT_DIR, file)
        df = pd.read_csv(file_path)

        # 先刪掉重複的 rows
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            print(f"{file}: 發現 {before - after} 筆重複資料 → 已刪除")

        # 找出要計算的特徵欄位
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # 計算總和並過濾
        df["row_sum"] = df[feature_cols].sum(axis=1)
        df_filtered = df[df["row_sum"] >= 50].drop(columns=["row_sum"])

        # 判斷是否有剩下資料
        if len(df_filtered) > 0:
            output_path = os.path.join(OUTPUT_DIR, file)
            df_filtered.to_csv(output_path, index=False)
            print(f"{file}: 原始 {before} 筆 → 去重 {after} 筆 → 過濾後 {len(df_filtered)} 筆 → 已輸出")
        else:
            print(f"{file}: 原始 {before} 筆 → 過濾後 0 筆 → 檔案刪除 (不輸出)")

print("全部處理完成！")

