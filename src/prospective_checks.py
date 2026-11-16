"""Sanity checks for the prospective test set: are any 'new' miRNAs aliases/duplicates of miRNAs known before
the cut-off (identical hairpin or identical mature sequence)? How similar are new miRNAs to known ones?"""
import numpy as np, pandas as pd
P = "data/processed_v34/"
A = np.load(P + "assoc.npy"); T = np.load(P + "first_pmid.npy"); cut = 29845432
mir = pd.read_csv(P + "mirnas.csv").mirna.values
fm = np.where(T > 0, T, np.iinfo(np.int64).max).min(1); new = fm > cut; old = ~new
Smt = np.load(P + "seqsim_mature_max.npy"); Shp = np.load(P + "seqsim_hairpin_max.npy")
hm = np.load(P + "has_mature.npy"); hh = np.load(P + "has_hairpin.npy")
iden_hp = (Shp[np.ix_(new, old)] >= 0.999) & hh[new][:, None] & hh[old][None, :]
iden_mt = (Smt[np.ix_(new, old)] >= 0.999) & hm[new][:, None] & hm[old][None, :]
newi, oldi = np.flatnonzero(new), np.flatnonzero(old)
rows = []
for k, i in enumerate(newi):
    j_hp = oldi[iden_hp[k]]; j_mt = oldi[iden_mt[k]]
    S = np.where(hm[i] & hm[oldi], Smt[i, oldi], Shp[i, oldi])
    rows.append(dict(mirna=mir[i], n_assoc=int(A[i].sum()), identical_hairpin_to=";".join(mir[j_hp]), identical_mature_to=";".join(mir[j_mt]),
                     max_identity_to_known=float(S.max())))
r = pd.DataFrame(rows); r.to_csv("results/prospective_new_mirna_checks.csv", index=False)
print("new miRNAs:", len(r), "| identical hairpin to a known miRNA:", int((r.identical_hairpin_to != "").sum()),
      "| identical mature to a known miRNA:", int((r.identical_mature_to != "").sum()))
print("max identity to known miRNAs: quantiles", r.max_identity_to_known.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
print("share with identity >= 0.8:", round((r.max_identity_to_known >= 0.8).mean(), 3), "| < 0.5:", round((r.max_identity_to_known < 0.5).mean(), 3))
print(r[r.identical_mature_to != ""].head(10).to_string())
