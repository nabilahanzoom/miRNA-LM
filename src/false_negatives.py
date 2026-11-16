"""How many test negatives later became reported associations (HMDD v4.0), and does removing them change AUROC?
Uses the saved cold-miRNA predictions (results/preds_degree) and the merged v3+v4 universe."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from temporal import temporal_split
P3, P34 = Path("data/processed"), Path("data/processed_v34")
m3 = pd.read_csv(P3 / "mirnas.csv").mirna.tolist(); d3 = pd.read_csv(P3 / "diseases.csv").disease.tolist()
m34 = pd.read_csv(P34 / "mirnas.csv").mirna.tolist(); d34 = pd.read_csv(P34 / "diseases.csv").disease.tolist()
A3 = np.load(P3 / "assoc.npy"); A34 = np.load(P34 / "assoc.npy")
mi = {m: i for i, m in enumerate(m34)}; di = {d: i for i, d in enumerate(d34)}
# future positive in v3 index space: positive in v3+v4 universe but not in v3
F = np.zeros_like(A3, bool)
for i, m in enumerate(m3):
    if m not in mi: continue
    for j, d in enumerate(d3):
        if d in di and A34[mi[m], di[d]] == 1 and A3[i, j] == 0:
            F[i, j] = True
print("v3 unknown pairs later reported in v4:", int(F.sum()), f"({F.sum() / (A3 == 0).sum():.4%} of all unknown pairs)")
rows = []
for f in glob.glob("results/preds_degree/cold_mirna_*_mlp_*.npz"):
    name = Path(f).stem.split("_"); neg = name[2]; feat = "_".join(name[4:-2]); seed, fold = name[-2], name[-1]
    z = np.load(f); pr, s, y = z["pairs"], z["score"], z["y"]
    fn = (y == 0) & F[pr[:, 0], pr[:, 1]]
    keep = ~fn
    rows.append(dict(neg=neg, feature=feat, seed=seed, fold=fold, n_neg=int((y == 0).sum()), n_false_neg=int(fn.sum()),
                     auc=roc_auc_score(y, s), auc_clean=roc_auc_score(y[keep], s[keep]),
                     auc_relabel=roc_auc_score(np.where(fn, 1, y), s)))
r = pd.DataFrame(rows)
r["frac_fn"] = r.n_false_neg / r.n_neg
s = r.groupby(["neg", "feature"])[["frac_fn", "auc", "auc_clean", "auc_relabel"]].mean().round(4)
print(s.to_string())
r.to_csv("results/false_negatives_cv.csv", index=False); s.to_csv("results/false_negatives_cv_summary.csv")
# within-v3 temporal split (cut-off 2014): fraction of test negatives reported later (2014-2018 in v3, or in v4)
T = np.load(P3 / "first_pmid.npy")
out = []
for seed in range(5):
    region, tests, cold, A_tr = temporal_split(A3, T, 25_000_000, seed)
    for k, (tp, tn) in tests.items():
        out.append(dict(seed=seed, test=k, n_neg=len(tn), frac_later_reported_v4=F[tn[:, 0], tn[:, 1]].mean()))
o = pd.DataFrame(out).groupby("test").mean(numeric_only=True).round(4)
print(o.to_string()); o.to_csv("results/false_negatives_temporal.csv")
