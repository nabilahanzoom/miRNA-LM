"""Aggregate results, paired tests and homology-stratified analysis.

  python analyze.py            -> results/summary_*.csv, results/homology_strata.csv
"""
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score

R = Path("results")
P = Path("data/processed")

frames = []
for f in glob.glob(str(R / "raw*.csv")):
    d = pd.read_csv(f)
    if "train_neg" not in d:
        d["train_neg"] = "uniform"
    frames.append(d)
df = pd.concat(frames).drop_duplicates(["setting", "model", "feature", "seed", "fold", "train_neg", "neg"])

pop = pd.read_csv(R / "popularity_baseline.csv").assign(model="none", feature="popularity", train_neg="none")
full = pd.concat([df, pop], ignore_index=True)
summ = (full.groupby(["setting", "neg", "train_neg", "model", "feature"])[["auc", "aupr"]]
        .agg(["mean", "std", "count"]).round(6))
summ.columns = ["_".join(c) for c in summ.columns]
summ.reset_index().to_csv(R / "summary_all.csv", index=False)


def paired(setting, neg, a, b, model="mlp", train_neg="degree", metric="auc"):
    key = ["seed", "fold"]
    x = df[(df.setting == setting) & (df.neg == neg) & (df.model == model) & (df.train_neg == train_neg)]
    xa = x[x.feature == a].set_index(key)[metric]
    xb = x[x.feature == b].set_index(key)[metric]
    j = xa.index.intersection(xb.index)
    if len(j) < 5:
        return None
    diff = xa[j] - xb[j]
    p = wilcoxon(xa[j], xb[j]).pvalue if (diff != 0).any() else 1.0
    return dict(setting=setting, neg=neg, model=model, train_neg=train_neg, a=a, b=b, n=len(j),
                mean_diff=diff.mean().round(4), wins=int((diff > 0).sum()), p=p)


tests = []
for tn in ["uniform", "degree"]:
    feats = df[(df.setting == "cold_mirna") & (df.train_neg == tn) & (df.model == "mlp")].feature.unique()
    for a, b in combinations(sorted(feats), 2):
        for neg in ["uniform", "degree"]:
            r = paired("cold_mirna", neg, a, b, train_neg=tn)
            if r:
                tests.append(r)
pd.DataFrame(tests).to_csv(R / "paired_tests.csv", index=False)

# ---------------- homology strata (cold-miRNA, saved predictions)
sim = np.load(P / "seqsim_mature_max.npy")
hp = np.load(P / "seqsim_hairpin_max.npy")
has_mt = np.load(P / "has_mature.npy")
S = np.where(has_mt[:, None] & has_mt[None, :], sim, hp)   # fall back to hairpin similarity
rows = []
for f in glob.glob(str(R / "preds_degree" / "cold_mirna_*_mlp_*.npz")):
    name = Path(f).stem                      # cold_mirna_<neg>_mlp_<feat>_s<seed>_f<fold>
    parts = name.split("_")
    neg, feat, seed, fold = parts[2], "_".join(parts[4:-2]), parts[-2], parts[-1]
    z = np.load(f)
    pairs, s, y = z["pairs"], z["score"], z["y"]
    test_m = np.unique(pairs[:, 0])
    train_m = np.setdiff1d(np.arange(S.shape[0]), test_m)
    nn = S[np.ix_(test_m, train_m)].max(1)
    nn_of = dict(zip(test_m, nn))
    h = np.array([nn_of[m] for m in pairs[:, 0]])
    for lab, lo, hi in [("low (<0.5)", 0, 0.5), ("mid (0.5-0.8)", 0.5, 0.8), ("high (>=0.8)", 0.8, 1.01)]:
        k = (h >= lo) & (h < hi)
        if k.sum() > 20 and 0 < y[k].mean() < 1:
            rows.append(dict(neg=neg, feature=feat, seed=seed, fold=fold, stratum=lab,
                             n_pairs=int(k.sum()), n_mirna=len(np.unique(pairs[k, 0])),
                             auc=roc_auc_score(y[k], s[k]), aupr=average_precision_score(y[k], s[k])))
if rows:
    hs = pd.DataFrame(rows)
    hs.to_csv(R / "homology_strata_raw.csv", index=False)
    (hs.groupby(["neg", "stratum", "feature"])[["auc", "n_mirna"]].agg(["mean", "std"]).round(4)
       .to_csv(R / "homology_strata.csv"))
print("done")
