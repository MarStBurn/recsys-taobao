"""Greedy gain-driven feature search over the generated space.

Each round every unselected candidate is scored by how much a tiny booster trained on that
one feature, stacked on the current model's out-of-fold predictions (init_score), lowers the
validation logloss. The best few are added, the model is refit, and the loop repeats.
Train = search users on day 6, validation = search users on day 7; the holdout users are
never touched here.

usage: python src/autofe.py [BUDGET]
"""
import json
import sys
import time
import multiprocessing as mp

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

import fe
from config import FEAT_DIR, ROOT, RESULTS, SEARCH_GROUP
from model import BASE_PARAMS

PER_ROUND = 8
MAX_CORR = 0.97
PROBE = dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=100,
             max_bin=63, verbose=-1, num_threads=1, seed=0)
PROBE_ROUNDS = 20
WORK = ROOT / "data/autofe_work"  # memory-mapped matrices shared with the probe workers


def load_prims(day, neg_frac, seed):
    df = pl.read_parquet(FEAT_DIR / f"g{SEARCH_GROUP}_d{day}.parquet")
    rnd = np.random.default_rng(seed).random(df.height)
    df = df.filter((pl.col("label") == 1) | pl.Series(rnd < neg_frac))
    cols = {c: df[c].to_numpy().astype(np.float32) for c in df.columns if "__" in c}
    return cols, df["label"].to_numpy()


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_base(Xtr, ytr, Xva, yva):
    """2-fold out-of-fold raw scores on train, averaged raw scores on validation."""
    if Xtr.shape[1] == 0:
        prior = logit(np.array([ytr.mean()]))[0]
        return np.full(len(ytr), prior), np.full(len(yva), prior), {}
    fold = np.random.default_rng(1).integers(0, 2, len(ytr))
    oof, va = np.zeros(len(ytr)), np.zeros(len(yva))
    params = {**BASE_PARAMS, "learning_rate": 0.1}
    for k in (0, 1):
        tr = fold != k
        m = lgb.train(params, lgb.Dataset(Xtr[tr], ytr[tr]), 3000,
                      valid_sets=[lgb.Dataset(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[~tr] = m.predict(Xtr[~tr], raw_score=True)
        va += m.predict(Xva, raw_score=True) / 2
    p = 1 / (1 + np.exp(-va))
    return oof, va, {"auc": roc_auc_score(yva, va), "logloss": log_loss(yva, p)}


_W = {}


def _worker_init():
    names = (WORK / "names.txt").read_text().split()
    for part in ("tr", "va"):
        M = np.load(WORK / f"{part}.npy", mmap_mode="r")
        _W[part] = {n: M[j] for j, n in enumerate(names)}
        _W["y" + part] = np.load(WORK / f"y{part}.npy")


def _worker_probe(args):
    spec, rnd = args
    if _W.get("rnd") != rnd:
        _W["init_tr"], _W["init_va"] = np.load(WORK / "init_tr.npy"), np.load(WORK / "init_va.npy")
        _W["base_ll"] = log_loss(_W["yva"], 1 / (1 + np.exp(-_W["init_va"])))
        _W["rnd"] = rnd
    gain, _ = probe(spec, _W["tr"], _W["va"], _W["ytr"], _W["yva"], _W["init_tr"], _W["init_va"], _W["base_ll"])
    return gain


def probe(spec, tr, va, ytr, yva, init_tr, init_va, base_ll):
    xtr = fe.compute(spec, tr)[:, None]
    xva = fe.compute(spec, va)[:, None]
    m = lgb.train(PROBE, lgb.Dataset(xtr, ytr, init_score=init_tr), PROBE_ROUNDS)
    raw = init_va + m.predict(xva, raw_score=True)
    return base_ll - log_loss(yva, 1 / (1 + np.exp(-raw))), xva[:, 0]


def main(budget: int):
    t0 = time.time()
    tr, ytr = load_prims(6, neg_frac=0.1, seed=0)
    va, yva = load_prims(7, neg_frac=0.2, seed=1)
    space = fe.search_space(sorted(tr))
    WORK.mkdir(parents=True, exist_ok=True)
    names = sorted(tr)
    (WORK / "names.txt").write_text("\n".join(names))
    np.save(WORK / "tr.npy", np.vstack([tr[n] for n in names]))  # feature-major: one contiguous row each
    np.save(WORK / "va.npy", np.vstack([va[n] for n in names]))
    np.save(WORK / "ytr.npy", ytr)
    np.save(WORK / "yva.npy", yva)
    pool = mp.get_context("spawn").Pool(10, initializer=_worker_init)
    print(f"train {len(ytr)} rows ({ytr.sum()} pos), valid {len(yva)} rows ({yva.sum()} pos), "
          f"space {len(space)} candidates, loaded in {time.time() - t0:.0f}s", flush=True)

    selected, trace, rounds = [], [], []
    rnd = 0
    while len(selected) < budget:
        Xtr = np.column_stack([fe.compute(s, tr) for s in selected]) if selected else np.zeros((len(ytr), 0))
        Xva = np.column_stack([fe.compute(s, va) for s in selected]) if selected else np.zeros((len(yva), 0))
        init_tr, init_va, base_m = fit_base(Xtr, ytr, Xva, yva)
        base_ll = log_loss(yva, 1 / (1 + np.exp(-init_va)))
        todo = [s for s in space if s not in selected]
        t = time.time()
        np.save(WORK / "init_tr.npy", init_tr)
        np.save(WORK / "init_va.npy", init_va)
        gains = pool.map(_worker_probe, [(s, rnd) for s in todo], chunksize=8)
        order = np.argsort([-g for g in gains])
        added = []
        for i in order:
            gain, col = gains[i], fe.compute(todo[i], va)
            if gain <= 0 or len(added) == PER_ROUND or len(selected) + len(added) >= budget:
                break
            # skip near-duplicates of what this round already picked
            if any(abs(np.corrcoef(np.nan_to_num(col), np.nan_to_num(c))[0, 1]) > MAX_CORR for _, c in added):
                continue
            added.append((todo[i], col))
            trace.append({"round": rnd, "spec": todo[i], "probe_gain": gain})
        print(f"[{time.time() - t0:5.0f}s] {len(selected):3d} feats  valid auc {base_m.get('auc', 0.5):.4f}"
              f"  probe {time.time() - t:.0f}s  +{[a for a, _ in added][:3]}...", flush=True)
        rounds.append({"round": rnd, "n_features": len(selected), **base_m})
        if not added:
            break
        selected += [a for a, _ in added]
        rnd += 1

    pool.close()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "autofe_selected.json").write_text(json.dumps({"selected": selected, "trace": trace, "rounds": rounds}, indent=1))
    print(f"done: {len(selected)} features in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 67)
