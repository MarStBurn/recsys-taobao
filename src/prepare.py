"""Convert raw UserBehavior.csv to Parquet, drop out-of-range timestamps, add day index.

The dataset covers 2017-11-25..2017-12-03 in China time (UTC+8). Day 0 = Nov 25.
"""
import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/UserBehavior.csv"
OUT = ROOT / "data/events.parquet"

START = 1511539200  # 2017-11-25 00:00 UTC+8
END = START + 9 * 86400

con = duckdb.connect()
con.execute(f"""
COPY (
  SELECT user_id, item_id, cat_id,
         CASE behavior WHEN 'pv' THEN 0 WHEN 'fav' THEN 1 WHEN 'cart' THEN 2 WHEN 'buy' THEN 3 END::TINYINT AS ev,
         ts,
         ((ts - {START}) // 86400)::TINYINT AS day
  FROM read_csv('{RAW}', header=false,
       columns={{'user_id':'INTEGER','item_id':'INTEGER','cat_id':'INTEGER','behavior':'VARCHAR','ts':'BIGINT'}})
  WHERE ts >= {START} AND ts < {END}
  ORDER BY user_id, ts
) TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)
""")
print(con.execute(f"""
  SELECT count(*) rows, count(DISTINCT user_id) users, count(DISTINCT item_id) items, count(DISTINCT cat_id) cats
  FROM '{OUT}'""").fetchall())
print(con.execute(f"SELECT day, count(*), sum((ev=3)::INT) buys FROM '{OUT}' GROUP BY 1 ORDER BY 1").fetchall())
print(con.execute(f"SELECT ev, count(*) FROM '{OUT}' GROUP BY 1 ORDER BY 1").fetchall())
