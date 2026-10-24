"""Build candidates, labels and the full primitive feature grid for one (user group, cutoff day).

A primitive is one aggregate over the event log strictly before the cutoff:
    {entity}__{event}__{agg}__{window_hours}
entity: u, i, c, ui, uc   event: pv, fav, cart, buy, any
agg:    cnt (count), rec (hours since last), nuniq (distinct items / users), ndays
u, ui, uc come from the sampled users; i and c come from every user.

usage: python src/build.py GROUP DAY
"""
import sys
import time
import duckdb
from config import (CAND_H, ENTITIES, EVENTS, EVENTS_MAP, FEAT_DIR, LOOKBACK_H, N_GROUPS,
                    WINDOWS, cutoff_ts)

KEYS = {"u": ["user_id"], "i": ["item_id"], "c": ["cat_id"],
        "ui": ["user_id", "item_id"], "uc": ["user_id", "cat_id"]}
# what nuniq counts for each entity; ui has none
NUNIQ = {"u": "item_id", "uc": "item_id", "i": "user_id", "c": "user_id"}


def entity_sql(ent: str, src: str, where: str, C: int) -> str:
    """One row per entity key with every primitive for that entity.

    Distinct counts avoid DISTINCT: an (entity, id) pair is active within a window
    exactly when its last event is inside it, so pairs are reduced to last timestamps first.
    """
    keys = ", ".join(KEYS[ent])
    cols = []
    for e, cond in EVENTS_MAP.items():
        for w in WINDOWS:
            cols.append(f"count(*) FILTER (WHERE {cond} AND ts >= {C - w * 3600}) AS {ent}__{e}__cnt__{w}")
        cols.append(f"({C} - max(ts) FILTER (WHERE {cond})) / 3600.0 AS {ent}__{e}__rec")
    parts = [f"SELECT {keys}, {', '.join(cols)} FROM {src} {where} GROUP BY ALL"]

    if ent in NUNIQ:
        other = NUNIQ[ent]
        ucols = [f"count(*) FILTER (WHERE last_{e} >= {C - w * 3600}) AS {ent}__{e}__nuniq__{w}"
                 for e in ("any", "buy") for w in WINDOWS]
        parts.append(f"""SELECT {keys}, {', '.join(ucols)} FROM (
            SELECT {keys}, {other}, max(ts) AS last_any, max(ts) FILTER (WHERE ev=3) AS last_buy
            FROM {src} {where} GROUP BY ALL) GROUP BY ALL""")

    day_c = (C - cutoff_ts(0)) // 86400
    dcols = [f"count(*) FILTER (WHERE has_{e} AND day >= {day_c - w // 24}) AS {ent}__{e}__ndays__{w}"
             for e in ("any", "buy") for w in (72, 144)]
    parts.append(f"""SELECT {keys}, {', '.join(dcols)} FROM (
        SELECT {keys}, day, TRUE AS has_any, bool_or(ev=3) AS has_buy
        FROM {src} {where} GROUP BY ALL) GROUP BY ALL""")

    sql = f"SELECT * FROM ({parts[0]}) p0"
    for k, part in enumerate(parts[1:], 1):
        sql += f" JOIN ({part}) p{k} USING ({keys})"
    return sql


def build(group: int, day: int) -> None:
    C = cutoff_ts(day)
    lo = C - LOOKBACK_H * 3600
    con = duckdb.connect(config={"memory_limit": "10GB", "preserve_insertion_order": False})
    con.execute(f"CREATE TEMP TABLE hist AS SELECT * FROM '{EVENTS}' WHERE ts >= {lo} AND ts < {C}")
    con.execute(f"CREATE TEMP TABLE hist_s AS SELECT * FROM hist WHERE user_id % {N_GROUPS} = {group}")
    con.execute(f"""
        CREATE TEMP TABLE cand AS
        SELECT user_id, item_id, min(cat_id) AS cat_id FROM hist_s WHERE ts >= {C - CAND_H * 3600} GROUP BY ALL""")
    con.execute(f"""
        CREATE TEMP TABLE cand_l AS
        SELECT c.*, (b.user_id IS NOT NULL)::TINYINT AS label
        FROM cand c LEFT JOIN (
            SELECT DISTINCT user_id, item_id FROM '{EVENTS}'
            WHERE day = {day} AND ev = 3 AND user_id % {N_GROUPS} = {group}) b USING (user_id, item_id)""")

    selects, joins = ["cand_l.*"], []
    for ent in ENTITIES:
        keys = KEYS[ent]
        src = "hist_s" if "u" in ent else "hist"
        restrict = " AND ".join(f"{k} IN (SELECT DISTINCT {k} FROM cand)" for k in keys if k != "user_id")
        where = f"WHERE {restrict}" if restrict else ""
        t = time.time()
        con.execute(f"CREATE TEMP TABLE t_{ent} AS {entity_sql(ent, src, where, C)}")
        print(f"  {ent}: {time.time() - t:.1f}s", flush=True)
        cols = [r[0] for r in con.execute(f"DESCRIBE t_{ent}").fetchall() if r[0] not in keys]
        for col in cols:
            # rec stays NULL when the event never happened; counts become 0
            selects.append(f"t_{ent}.{col}::FLOAT AS {col}" if col.endswith("__rec")
                           else f"coalesce(t_{ent}.{col}, 0)::FLOAT AS {col}")
        joins.append(f"LEFT JOIN t_{ent} USING ({', '.join(keys)})")

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out = FEAT_DIR / f"g{group}_d{day}.parquet"
    con.execute(f"COPY (SELECT {', '.join(selects)} FROM cand_l {' '.join(joins)} ORDER BY user_id)"
                f" TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    n, pos, users = con.execute("SELECT count(*), sum(label), count(DISTINCT user_id) FROM cand_l").fetchone()
    print(f"g{group} d{day}: {n} candidates, {pos} positives, {users} users, {len(selects) - 1 - 4} features -> {out.name}")


if __name__ == "__main__":
    build(int(sys.argv[1]), int(sys.argv[2]))
