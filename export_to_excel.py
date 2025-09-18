import os, math
import pandas as pd
from sqlalchemy import create_engine, text

DB = "logplayerplugin_prod"
HOST = "127.0.0.1"
USER = "root"
PASSWORD = "Ff0905593231"
OUTDIR = r"C:\\Users\\hsu96\\OneDrive\\Desktop\\mysql_csv_exports\\excel\\logplayerplugin_prod"

os.makedirs(OUTDIR, exist_ok=True)

# 用 mysql-connector 當 SQLAlchemy driver；也可改成 pymysql（mysql+pymysql://...）
engine = create_engine(f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}/{DB}?charset=utf8mb4")

MAX_ROWS_PER_FILE = 1_000_000

with engine.begin() as conn:
    # 取所有資料表
    tables = [row[0] for row in conn.execute(text("SHOW TABLES")).fetchall()]

    for t in tables:
        # 計數
        total = conn.execute(text(f"SELECT COUNT(*) FROM `{t}`")).scalar()
        parts = max(1, math.ceil(total / MAX_ROWS_PER_FILE))
        offset = 0
        if total == 1:
            print(f"Skip {t}: rows=1")
            continue
        for p in range(1, parts + 1):
            limit = min(MAX_ROWS_PER_FILE, total - offset)
            print(f"Exporting {t} part {p}/{parts} rows {limit}...")

            # 讀資料（參數化避免 SQL injection，雖然這裡用不到）
            df = pd.read_sql(
                text(f"SELECT * FROM `{t}` LIMIT :limit OFFSET :offset"),
                conn,
                params={"limit": limit, "offset": offset}
            )

            out = os.path.join(OUTDIR, f"{t}_part{p}.csv" if parts > 1 else f"{t}.csv")
            df.to_csv(out, index=False, encoding="utf-8-sig")
            offset += limit

print("Done.")
