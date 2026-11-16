"""How 'novel' are test miRNAs? Cross-validated cold-miRNA folds (HMDD v3.0) vs prospective new miRNAs (v4.0)."""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
def ident(P):
    hm = np.load(P + "has_mature.npy")
    return np.where(hm[:, None] & hm[None, :], np.load(P + "seqsim_mature_max.npy"), np.load(P + "seqsim_hairpin_max.npy"))
rows = []
P = "data/processed/"; S = ident(P); g = pd.read_csv(P + "mirna_family.csv")
for seed in range(3):
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(g):
        nn = S[np.ix_(te, tr)].max(1)
        rows.append(dict(test="cross-validation (cold-miRNA)", nn_median=np.median(nn), frac_ge_0_8=(nn >= 0.8).mean(), frac_lt_0_5=(nn < 0.5).mean(),
                         seed_in_train=g.seed.iloc[te].isin(set(g.seed.iloc[tr].dropna())).mean(),
                         namefam_in_train=g.name_family.iloc[te].isin(set(g.name_family.iloc[tr])).mean()))
P = "data/processed_v34/"; S = ident(P); g = pd.read_csv(P + "mirna_family.csv")
T = np.load(P + "first_pmid.npy"); fm = np.where(T > 0, T, np.iinfo(np.int64).max).min(1); new = np.flatnonzero(fm > 29845432); old = np.flatnonzero(fm <= 29845432)
nn = S[np.ix_(new, old)].max(1)
rows.append(dict(test="prospective (new in HMDD v4.0)", nn_median=np.median(nn), frac_ge_0_8=(nn >= 0.8).mean(), frac_lt_0_5=(nn < 0.5).mean(),
                 seed_in_train=g.seed.iloc[new].isin(set(g.seed.iloc[old].dropna())).mean(),
                 namefam_in_train=g.name_family.iloc[new].isin(set(g.name_family.iloc[old])).mean()))
r = pd.DataFrame(rows).groupby("test", sort=False).mean().round(3); r.to_csv("results/novelty_comparison.csv"); print(r.to_string())
