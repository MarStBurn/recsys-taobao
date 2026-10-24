from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/events.parquet"
FEAT_DIR = ROOT / "data/feat"
RESULTS = ROOT / "results"

START = 1511539200  # 2017-11-25 00:00 UTC+8, day 0
LOOKBACK_H = 144    # every feature sees at most 6 days before the cutoff
CAND_H = 72         # candidates: items touched in the 3 days before the cutoff

# Feature-search users train on day 6 and validate on day 7.
# Holdout users are only ever scored on day 8.
SEARCH_GROUP = 0
HOLDOUT_GROUP = 5
N_GROUPS = 10

WINDOWS = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144]
EVENTS_MAP = {"pv": "ev=0", "fav": "ev=1", "cart": "ev=2", "buy": "ev=3", "any": "TRUE"}
ENTITIES = ["u", "i", "c", "ui", "uc"]


def cutoff_ts(day: int) -> int:
    return START + day * 86400
