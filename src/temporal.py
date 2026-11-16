"""Temporal validation using first-evidence PMID as a time stamp (PMIDs are assigned ~chronologically).

Training: associations first reported with PMID <= CUT (~mid-2014 for 25,000,000).
Test 1 'tcold': miRNAs whose first disease association appears after CUT (emerging miRNAs) -> all their pairs.
Test 2 'twarm': associations first reported after CUT for miRNAs already known before CUT.
Future associations are never used as training negatives. Each test has uniform and
popularity-matched negatives (same rules as experiment.py).

  python src/temporal.py --cut 25000000 --features ... --seeds 0 1 2 3 4 --train_neg degree --out results/temporal_degree.csv
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
from experiment import load, run_fold, _degree_matched_neg


def temporal_split(A, T, cut, seed):
    rng = np.random.default_rng(seed)
    fm = np.where(T > 0, T, np.iinfo(np.int64).max).min(1)
    new = fm > cut                                    # emerging miRNAs
    future = (T > cut)
    A_tr = A * (~future) * (~new[:, None])
    deg_d = A_tr.sum(0) + 1
    # tcold
    sub = np.zeros_like(A, bool); sub[new] = True
    tp_c = np.argwhere((A == 1) & sub)
    cand = np.argwhere((A == 0) & sub)
    tn_cu = cand[rng.choice(len(cand), len(tp_c), replace=False)]
    ok = sub.copy(); ok[tn_cu[:, 0], tn_cu[:, 1]] = False
    tn_cd = _degree_matched_neg(A, tp_c, ok, 1, deg_d, rng)
    # twarm
    old = ~new
    tp_w = np.argwhere(future & old[:, None])
    cand = np.argwhere((A == 0) & old[:, None])
    tn_wu = cand[rng.choice(len(cand), len(tp_w), replace=False)]
    ok = np.zeros_like(A, bool); ok[old] = True; ok[tn_wu[:, 0], tn_wu[:, 1]] = False
    tn_wd = _degree_matched_neg(A, tp_w, ok, 1, deg_d, rng)
    region = np.zeros_like(A, bool); region[old] = True
    for t in (tp_w, tn_wu, tn_wd):
        region[t[:, 0], t[:, 1]] = False
    assert (A * region == A_tr).all()
    tests = {"tcold_uniform": (tp_c, tn_cu), "tcold_degree": (tp_c, tn_cd),
             "twarm_uniform": (tp_w, tn_wu), "twarm_degree": (tp_w, tn_wd)}
    return region, tests, (0, new), A_tr


def popularity(A_tr, tests, seed):
    dm, dd = A_tr.sum(1), A_tr.sum(0); rows = []
    for mode, (tp, tn) in tests.items():
        te = np.concatenate([tp, tn]); y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]
        s = dd[te[:, 1]] if mode.startswith("tcold") else np.log1p(dm[te[:, 0]]) + np.log1p(dd[te[:, 1]])
        s = s + 1e-9 * np.random.default_rng(seed).random(len(s))
        rows.append(dict(model="none", feature="popularity", seed=seed, train_neg="none", neg=mode,
                         auc=roc_auc_score(y, s), aupr=average_precision_score(y, s), n_test=len(y)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed"); ap.add_argument("--out", required=True)
    ap.add_argument("--cut", type=int, default=25_000_000)
    ap.add_argument("--features", nargs="+", required=True); ap.add_argument("--models", nargs="+", default=["mlp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--train_neg", default="degree"); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--popularity", action="store_true")
    ap.add_argument("--save_preds", default="")
    a = ap.parse_args()
    D = load(a.data); A = D["A"]; T = np.load(Path(a.data) / "first_pmid.npy")
    out = Path(a.out); done = set()
    if out.exists():
        done = set(map(tuple, pd.read_csv(out)[["model", "feature", "seed"]].astype(str).values))
    for seed in a.seeds:
        region, tests, cold, A_tr = temporal_split(A, T, a.cut, seed)
        if seed == a.seeds[0]:
            print({k: (len(v[0]), len(v[1])) for k, v in tests.items()}, "train pos", int(A_tr.sum()), flush=True)
        if a.popularity:
            rows = [dict(cut=a.cut, **r) for r in popularity(A_tr, tests, seed)]
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False); continue
        for model in a.models:
            for feat in a.features:
                if (model, feat, str(seed)) in done:
                    continue
                t0 = time.time()
                res = run_fold(D, A, region, None, tests, cold, feat, model, seed, a.epochs, a.train_neg)
                rows = [dict(cut=a.cut, model=model, feature=feat, seed=seed, train_neg=a.train_neg, neg=k,
                             auc=v[0], aupr=v[1], n_test=len(v[4])) for k, v in res.items()]
                if a.save_preds:
                    pp = Path(a.save_preds); pp.mkdir(parents=True, exist_ok=True)
                    for k, v in res.items():
                        np.savez_compressed(pp / f"{k}_{model}_{feat}_{a.train_neg}_s{seed}.npz", pairs=v[2], score=v[3], y=v[4])
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(model, feat, seed, " ".join(f"{r['neg']}={r['auc']:.3f}" for r in rows), f"{time.time()-t0:.0f}s", flush=True)
