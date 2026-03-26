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

        # delete repeat rows
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)


        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # add total and output
        df["row_sum"] = df[feature_cols].sum(axis=1)
        df_filtered = df[df["row_sum"] >= 50].drop(columns=["row_sum"])


        if len(df_filtered) > 0:
            output_path = os.path.join(OUTPUT_DIR, file)
            df_filtered.to_csv(output_path, index=False)

print("Done")

