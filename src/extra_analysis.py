"""(1) paired tests vs. the popularity baseline; (2) how much of each miRNA representation
predicts miRNA research popularity (log degree), 5-fold ridge regression, label-only target."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import glob

df = pd.concat([pd.read_csv(f) for f in glob.glob("results/raw*.csv")]).drop_duplicates(
    ["setting", "model", "feature", "seed", "fold", "train_neg", "neg"])
pop = pd.read_csv("results/popularity_baseline.csv")
rows = []
for tn in ["uniform", "degree"]:
    for neg in ["uniform", "degree"]:
        p = pop[(pop.setting == "cold_mirna") & (pop.neg == neg)].set_index(["seed", "fold"]).auc
        x = df[(df.setting == "cold_mirna") & (df.model == "mlp") & (df.train_neg == tn) & (df.neg == neg)]
        for f, g in x.groupby("feature"):
            a = g.set_index(["seed", "fold"]).auc
            j = a.index.intersection(p.index)
            if len(j) < 15:
                continue
            d = a[j] - p[j]
            rows.append(dict(train_neg=tn, neg=neg, feature=f, n=len(j), auc=a[j].mean().round(4),
                             pop_auc=p[j].mean().round(4), diff=d.mean().round(4),
                             wins=int((d > 0).sum()), p=wilcoxon(a[j], p[j]).pvalue))
pd.DataFrame(rows).to_csv("results/paired_vs_popularity.csv", index=False)
print(pd.DataFrame(rows).to_string())

P = "data/processed/"
A = np.load(P + "assoc.npy")
mir = pd.read_csv(P + "mirnas.csv")
nm = len(mir)
if A.ndim == 2 and A.shape[1] == 2:
    deg = np.bincount(A[:, 0], minlength=nm)
else:
    deg = A.sum(1)
y = np.log1p(deg)
feats = {"random": np.random.default_rng(0).normal(size=(nm, 640)),
         "kmer": np.load(P + "kmer.npy"), "align": np.load(P + "seqsim_align.npy")}
for lm in ["rnafm", "ernierna", "rinalmo"]:
    feats[lm + "_both"] = np.hstack([np.load(P + f"emb_{lm}_mature.npy"), np.load(P + f"emb_{lm}_hairpin.npy")])
out = []
for name, X in feats.items():
    pred = np.zeros(nm)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13))).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out.append(dict(feature=name, spearman=round(spearmanr(pred, y)[0], 3), r2=round(r2, 3)))
pd.DataFrame(out).to_csv("results/popularity_encoding.csv", index=False)
print(pd.DataFrame(out).to_string())
