# Automated feature engineering for RecSys — Taobao UserBehavior

Work in progress. Full report with results is coming.

Can the hand-crafted feature step of a gradient-boosting recommender be automated?
A feature grammar (entity × event × window × aggregation, plus generated ratios) produces
1,453 candidate features; a greedy, gain-driven search selects among them, and the result is
compared against a tuned hand-crafted baseline on users never seen during the search.

## Reproduce

Download `UserBehavior.csv.zip` from [Tianchi (dataId 649)](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)
and unzip into `data/raw/`. macOS needs `brew install libomp` for LightGBM.

```
uv sync
uv run python src/prepare.py        # CSV -> parquet, day index
uv run python src/build.py 0 6      # search users, cutoff day 6 (also 0 7, 0 8, 5 8)
uv run python src/autofe.py 120     # automated feature search
uv run python src/final.py          # tune, retrain, score holdout users on day 8
```
