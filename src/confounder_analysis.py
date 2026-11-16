"""Does the popularity signal in RNA-LM embeddings go beyond discovery order / conservation / length?
5-fold CV ridge, target = log(1 + miRNA degree).  Reports
 (i)  Spearman of each covariate with popularity,
 (ii) rho of covariates-only, representation-only and covariates+representation models,
 (iii) rho of each representation for the popularity residual after removing covariates (within CV)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
P = "data/processed/"
A = np.load(P + "assoc.npy"); y = np.log1p(A.sum(1)); n = len(y)
c = pd.read_csv(P + "mirna_covariates.csv")
cov = pd.DataFrame(index=c.index)
for k in ["acc_hairpin", "acc_mature"]:
    cov[k] = np.log(c[k].fillna(c[k].median()))
for k in ["cons_seq", "cons_seed"]:
    cov[k] = np.log1p(c[k].fillna(0))
for k in ["len_mature", "len_hairpin"]:
    cov[k] = c[k].fillna(c[k].median())
cov["has_mature"] = c.acc_mature.notna().astype(float); cov["has_hairpin"] = c.acc_hairpin.notna().astype(float)
cov["n_records"] = c.n_records
print("Spearman of covariates with log degree:")
uni = {k: round(spearmanr(cov[k], y)[0], 3) for k in cov}
print(uni)
C = cov.values
F = {"random": np.random.default_rng(0).normal(size=(n, 640)), "kmer": np.load(P + "kmer.npy"),
     "align": np.load(P + "seqsim_align.npy")}
for lm in ["rnafm", "ernierna", "rinalmo"]:
    F[lm + "_both"] = np.hstack([np.load(P + f"emb_{lm}_mature.npy"), np.load(P + f"emb_{lm}_hairpin.npy")])
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
def cv(X, target, seeds=range(5)):
    rs = []
    for s in seeds:
        pred = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=s).split(X):
            pred[te] = ridge().fit(X[tr], target[tr]).predict(X[te])
        rs.append(spearmanr(pred, target)[0])
    return np.mean(rs), np.std(rs)
def cv_resid(X, seeds=range(5)):
    """residualise y on covariates inside each training fold, predict residual of the test fold"""
    rs = []
    for s in seeds:
        pr, rr = np.zeros(n), np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=s).split(X):
            lr = make_pipeline(StandardScaler(), LinearRegression()).fit(C[tr], y[tr])
            r_tr = y[tr] - lr.predict(C[tr]); rr[te] = y[te] - lr.predict(C[te])
            pr[te] = ridge().fit(X[tr], r_tr).predict(X[te])
        rs.append(spearmanr(pr, rr)[0])
    return np.mean(rs), np.std(rs)
rows = [dict(feature="covariates only", rho=cv(C, y)[0])]
for k, X in F.items():
    a, b = cv(X, y), cv(np.hstack([X, C]), y); r = cv_resid(X)
    rows.append(dict(feature=k, rho=a[0], rho_sd=a[1], rho_with_cov=b[0], rho_residual=r[0], rho_residual_sd=r[1]))
out = pd.DataFrame(rows).round(3)
out.to_csv("results/popularity_confounders.csv", index=False)
pd.Series(uni).to_csv("results/covariate_spearman.csv")
print(out.to_string(index=False))
