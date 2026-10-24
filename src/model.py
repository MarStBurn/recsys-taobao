"""Data loading, LightGBM training, and ranking metrics."""
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from config import FEAT_DIR
import fe

KEYS = ["user_id", "item_id", "cat_id", "label"]


def load(group: int, day: int, specs: list[str], neg_frac: float = 1.0, seed: int = 0):
    """Returns (X float32 [n, len(specs)], frame with keys + label)."""
    prims = sorted({p for s in specs for p in fe.needs(s)})
    df = pl.read_parquet(FEAT_DIR / f"g{group}_d{day}.parquet", columns=KEYS + prims)
    if neg_frac < 1.0:
        rnd = np.random.default_rng(seed).random(df.height)
        df = df.filter((pl.col("label") == 1) | pl.Series(rnd < neg_frac))
    cols = {p: df[p].to_numpy().astype(np.float32) for p in prims}
    X = np.column_stack([fe.compute(s, cols) for s in specs]) if specs else np.zeros((df.height, 0), np.float32)
    return X, df.select(KEYS)


BASE_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   verbose=-1, num_threads=10, seed=0)


def train(Xtr, ytr, Xva=None, yva=None, params=None, rounds=2000, init_tr=None, init_va=None):
    p = {**BASE_PARAMS, **(params or {})}
    dtr = lgb.Dataset(Xtr, ytr, init_score=init_tr, free_raw_data=False)
    if Xva is None:
        return lgb.train(p, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(Xva, yva, init_score=init_va, reference=dtr)
    return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def per_user_metrics(keys: pl.DataFrame, score: np.ndarray, ks=(5, 10, 20)) -> pl.DataFrame:
    """One row per user with >= 1 positive candidate: ndcg@k, recall@k, hit@k."""
    df = keys.select("user_id", "label").with_columns(score=pl.Series(score))
    df = df.filter(pl.col("label").sum().over("user_id") > 0)
    df = df.sort(["user_id", "score"], descending=[False, True]).with_columns(
        rank=pl.int_range(1, pl.len() + 1).over("user_id"))
    df = df.with_columns(gain=pl.col("label") / (pl.col("rank") + 1).log(2))
    agg = [pl.col("label").sum().alias("npos")]
    for k in ks:
        agg += [pl.col("gain").filter(pl.col("rank") <= k).sum().alias(f"dcg{k}"),
                pl.col("label").filter(pl.col("rank") <= k).sum().alias(f"hits{k}")]
    u = df.group_by("user_id").agg(agg)
    out = [pl.col("user_id")]
    for k in ks:
        idcg = pl.int_ranges(1, pl.min_horizontal("npos", pl.lit(k)) + 1).list.eval(
            1 / (pl.element() + 1).log(2)).list.sum()
        out += [(pl.col(f"dcg{k}") / idcg).alias(f"ndcg@{k}"),
                (pl.col(f"hits{k}") / pl.col("npos")).alias(f"recall@{k}"),
                (pl.col(f"hits{k}") > 0).cast(pl.Float64).alias(f"hit@{k}")]
    return u.select(out).sort("user_id")


def evaluate(keys: pl.DataFrame, score: np.ndarray, prob: np.ndarray | None = None) -> dict:
    y = keys["label"].to_numpy()
    m = {"auc": roc_auc_score(y, score)}
    if prob is not None:
        m["logloss"] = log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7))
    pu = per_user_metrics(keys, score)
    m.update({c: pu[c].mean() for c in pu.columns if c != "user_id"})
    m["users_with_pos"] = pu.height
    return m
