"""Tune, retrain and score each feature set on the holdout users, day 8.

For every set: random search over LightGBM params (train search users day 6, validate day 7),
retrain on days 6+7 with the best params, score holdout users on day 8 (and search users on
day 8 as a secondary check). Negatives are subsampled for training and the scores are shifted
back by log(NEG_FRAC) so logloss stays calibrated.

usage: python src/final.py
"""
import json
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

import fe
from config import HOLDOUT_GROUP, RESULTS, SEARCH_GROUP
from model import evaluate, load, per_user_metrics, train

NEG_FRAC = 0.3
N_TRIALS = 8
N_BOOT = 300
GRID = dict(num_leaves=[15, 31, 63, 127, 255], min_data_in_leaf=[50, 100, 200, 500, 1000],
            feature_fraction=[0.3, 0.5, 0.7, 0.9], lambda_l2=[0, 1, 10, 50], max_bin=[63, 255])


def feature_sets():
    auto = json.loads((RESULTS / "autofe_selected.json").read_text())["selected"]
    base = fe.baseline_specs()
    sets = {"baseline (hand-crafted)": base}
    for k in (16, 32):
        sets[f"auto top-{k}"] = auto[:k]
    sets[f"auto all ({len(auto)})"] = auto
    sets["baseline + auto"] = base + [s for s in auto if s not in base]
    return sets


def tune(specs):
    Xtr, ktr = load(SEARCH_GROUP, 6, specs, NEG_FRAC, seed=0)
    Xva, kva = load(SEARCH_GROUP, 7, specs)
    rng = np.random.default_rng(0)
    trials = [{}] + [{k: v[rng.integers(len(v))] for k, v in GRID.items()} for _ in range(N_TRIALS - 1)]
    best = None
    for p in trials:
        m = train(Xtr, ktr["label"].to_numpy(), Xva, kva["label"].to_numpy(), params=p)
        auc = roc_auc_score(kva["label"].to_numpy(), m.predict(Xva, raw_score=True))
        if best is None or auc > best[0]:
            best = (auc, p, m.best_iteration)
    return best


def boot_ci(a, b, n=N_BOOT):
    """Paired bootstrap over users of mean(a - b); a, b aligned per user."""
    d = a - b
    rng = np.random.default_rng(0)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def auc_boot_ci(keys, sa, sb, n=N_BOOT):
    """Paired user-cluster bootstrap of AUC(a) - AUC(b)."""
    users = keys["user_id"].to_numpy()
    y = keys["label"].to_numpy()
    uniq, inv = np.unique(users, return_inverse=True)
    order = np.argsort(inv)
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(inv))
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[u]:ends[u]] for u in pick])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    RESULTS.mkdir(exist_ok=True)
    rows, scores = [], {}
    for name, specs in feature_sets().items():
        t = time.time()
        val_auc, params, iters = tune(specs)
        X6, k6 = load(SEARCH_GROUP, 6, specs, NEG_FRAC, seed=0)
        X7, k7 = load(SEARCH_GROUP, 7, specs, NEG_FRAC, seed=1)
        m = train(np.vstack([X6, X7]), np.concatenate([k6["label"], k7["label"]]), params=params,
                  rounds=int(iters * 1.1))
        del X6, X7
        row = {"set": name, "n_features": len(specs), "val_auc_d7": val_auc, "params": params, "iters": iters}
        for tag, g in (("holdout", HOLDOUT_GROUP), ("search", SEARCH_GROUP)):
            X, k = load(g, 8, specs)
            raw = m.predict(X, raw_score=True) + np.log(NEG_FRAC)
            row[tag] = evaluate(k, raw, 1 / (1 + np.exp(-raw)))
            if tag == "holdout":
                scores[name] = (k, raw)
        rows.append(row)
        h = row["holdout"]
        print(f"{name:28s} n={len(specs):3d}  val auc {val_auc:.4f}  | holdout d8: auc {h['auc']:.4f} "
              f"logloss {h['logloss']:.5f} ndcg@10 {h['ndcg@10']:.4f} recall@10 {h['recall@10']:.4f}"
              f"  ({time.time() - t:.0f}s)", flush=True)

    # paired comparisons against the baseline on the holdout users
    bk, bs = scores["baseline (hand-crafted)"]
    bu = per_user_metrics(bk, bs)
    for row in rows:
        if row["set"].startswith("baseline (hand"):
            continue
        k, s = scores[row["set"]]
        u = per_user_metrics(k, s)
        assert (u["user_id"] == bu["user_id"]).all()
        row["vs_baseline"] = {m: boot_ci(u[m].to_numpy(), bu[m].to_numpy())
                              for m in ("ndcg@10", "recall@10", "ndcg@5", "recall@20")}
        row["vs_baseline"]["auc_ci"] = auc_boot_ci(k, s, bs)
        print(f"  {row['set']:28s} vs baseline: " + ", ".join(
            f"{m} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for m, v in row["vs_baseline"].items() if m != "auc_ci")
            + f", auc CI [{row['vs_baseline']['auc_ci'][0]:+.4f}, {row['vs_baseline']['auc_ci'][1]:+.4f}]",
            flush=True)
    (RESULTS / "final.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
