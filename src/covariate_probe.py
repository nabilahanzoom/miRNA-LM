"""How well does each representation predict miRNA age (log miRBase accession) and conservation?"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
P = "data/processed/"
c = pd.read_csv(P + "mirna_covariates.csv"); n = len(c)
F = {"random": np.random.default_rng(0).normal(size=(n, 640)), "kmer": np.load(P + "kmer.npy"),
     "align": np.load(P + "seqsim_align.npy")}
for lm in ["rnafm", "ernierna", "rinalmo"]:
    F[lm + "_both"] = np.hstack([np.load(P + f"emb_{lm}_mature.npy"), np.load(P + f"emb_{lm}_hairpin.npy")])
T = {"age (log hairpin accession)": (np.log(c.acc_hairpin), c.acc_hairpin.notna()),
     "conservation (log1p species, identical mature)": (np.log1p(c.cons_seq), c.cons_seq.notna())}
rows = []
for k, X in F.items():
    r = dict(feature=k)
    for t, (y, ok) in T.items():
        idx = np.where(ok)[0]; yy = y.values[idx]; XX = X[idx]; rs = []
        for s in range(5):
            pred = np.zeros(len(idx))
            for tr, te in KFold(5, shuffle=True, random_state=s).split(XX):
                pred[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13))).fit(XX[tr], yy[tr]).predict(XX[te])
            rs.append(spearmanr(pred, yy)[0])
        r[t] = round(np.mean(rs), 3)
    rows.append(r)
o = pd.DataFrame(rows); o.to_csv("results/covariate_probe.csv", index=False); print(o.to_string(index=False))
