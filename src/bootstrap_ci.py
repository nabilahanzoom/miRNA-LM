"""miRNA-level paired bootstrap CIs for AUROC differences (cold-miRNA CV, saved predictions).
For each bootstrap replicate, test miRNAs are resampled with replacement within every fold (all pairs of a
sampled miRNA are kept), fold AUROCs are recomputed and averaged over 5 folds x seeds. Also reports
equivalence (TOST-style: 90% CI inside +-margin) and Holm-adjusted Wilcoxon p-values.
  python src/bootstrap_ci.py results/preds_degree cold_mirna_{neg}_mlp_{feat}_s{seed}_f{fold}.npz out_prefix"""
import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
PRED, PAT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
A = np.load("data/processed/assoc.npy")
import os
FEATS = os.environ.get("FEATS", "ernierna_both rnafm_both rinalmo_both align kmer random").split()
COMPS = [("ernierna_both", "align"), ("rnafm_both", "align"), ("rinalmo_both", "align"),
         ("ernierna_both", "random"), ("align", "random"), ("ernierna_both", "popularity"), ("align", "popularity")]
COMPS = [c for c in COMPS if all(x in FEATS + ["popularity"] for x in c)]
rng = np.random.default_rng(0)

def load_fold(neg, feat, seed, fold):
    f = Path(PRED) / PAT.format(neg=neg, feat=feat, seed=seed, fold=fold)
    if not f.exists(): return None
    z = np.load(f); return z["pairs"], z["score"], z["y"]

def fast_auc(y, s):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = sums[inv] / cnt[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

rows, ci_rows = [], []
for neg in ["uniform", "degree", "exact"]:
    folds = []
    for seed in range(10):
        for fold in range(5):
            data = {f: load_fold(neg, f, seed, fold) for f in FEATS}
            if any(v is None for v in data.values()): continue
            pairs, _, y = data[FEATS[0]]
            for f in FEATS: assert (data[f][0] == pairs).all()
            held = np.unique(pairs[:, 0]); A_tr = A.copy(); A_tr[held] = 0
            data["popularity"] = (pairs, A_tr.sum(0)[pairs[:, 1]] + 1e-9 * np.random.default_rng(seed).random(len(y)), y)
            idx_by_m = {m: np.flatnonzero(pairs[:, 0] == m) for m in held}
            folds.append((seed, fold, data, y, held, idx_by_m))
    if not folds: continue
    allf = FEATS + ["popularity"]
    point = {f: np.mean([fast_auc(y, d[f][1]) for _, _, d, y, _, _ in folds]) for f in allf}
    per_fold = {f: np.array([fast_auc(y, d[f][1]) for _, _, d, y, _, _ in folds]) for f in allf}
    boot = {f: np.zeros(NB) for f in allf}
    for b in range(NB):
        acc = {f: [] for f in allf}
        for _, _, d, y, held, idx_by_m in folds:
            ms = rng.choice(held, len(held), replace=True)
            ii = np.concatenate([idx_by_m[m] for m in ms])
            yy = y[ii]
            if yy.min() == yy.max(): continue
            for f in allf: acc[f].append(fast_auc(yy, d[f][1][ii]))
        for f in allf: boot[f][b] = np.mean(acc[f])
    for f in allf:
        lo, hi = np.percentile(boot[f], [2.5, 97.5])
        rows.append(dict(neg=neg, feature=f, auc=point[f], ci_low=lo, ci_high=hi, n_folds=len(folds)))
    for a_, b_ in COMPS:
        diff = boot[a_] - boot[b_]; dpt = point[a_] - point[b_]
        lo, hi = np.percentile(diff, [2.5, 97.5]); lo90, hi90 = np.percentile(diff, [5, 95])
        pw = wilcoxon(per_fold[a_], per_fold[b_]).pvalue if np.any(per_fold[a_] != per_fold[b_]) else 1.0
        ci_rows.append(dict(neg=neg, a=a_, b=b_, diff=dpt, ci_low=lo, ci_high=hi, ci90_low=lo90, ci90_high=hi90,
                            p_boot=2 * min((diff <= 0).mean(), (diff >= 0).mean()), p_wilcoxon=pw,
                            equivalent_0_01=bool(lo90 > -0.01 and hi90 < 0.01)))
c = pd.DataFrame(ci_rows)
# Holm correction within each negative mode
c["p_holm"] = np.nan
for neg, g in c.groupby("neg"):
    order = g.p_wilcoxon.sort_values().index; m = len(order); adj = []; run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (m - k) * c.loc[i, "p_wilcoxon"])); c.loc[i, "p_holm"] = run
pd.DataFrame(rows).round(4).to_csv(f"{OUT}_auc_ci.csv", index=False); c.round(4).to_csv(f"{OUT}_diff_ci.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string()); print(c.round(4).to_string())
