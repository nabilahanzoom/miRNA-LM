"""Robustness of the prospective (HMDD v3.0 -> v4.0) test on newly reported miRNAs, from saved predictions.
  * miRNA-level bootstrap CIs (resample new miRNAs; AUROC averaged over 10 seeds) + Holm + equivalence
  * sensitivity: exclude new miRNAs whose hairpin or mature sequence is identical to a known miRNA (aliases)
  * stratification by nearest-homolog identity, seed family and name family present in training
  python src/prospective_robust.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

D = Path("data/processed_v34"); P = Path("results/preds_prosp"); CUT = 29845432; NB = 1000
FEATS = ["ernierna_both", "rnafm_both", "rinalmo_both", "align", "random"]
COMPS = [("ernierna_both", "random"), ("ernierna_both", "align"), ("ernierna_both", "popularity"),
         ("rnafm_both", "random"), ("rinalmo_both", "random"), ("align", "random"), ("align", "popularity")]
A = np.load(D / "assoc.npy"); T = np.load(D / "first_pmid.npy")
mir = pd.read_csv(D / "mirnas.csv").mirna.values
fm = np.where(T > 0, T, np.iinfo(np.int64).max).min(1); new = fm > CUT
A_tr = A * (T <= CUT) * (~new[:, None]); deg_d = A_tr.sum(0)

def auc(y, s):
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = np.empty(len(s)); r[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    r = (np.bincount(inv, weights=r) / cnt)[inv]; n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# per-miRNA attributes of the new miRNAs
chk = pd.read_csv("results/prospective_new_mirna_checks.csv").set_index("mirna")
fam = pd.read_csv(D / "mirna_family.csv").set_index("mirna")
known = set(mir[~new]); seeds_tr = set(fam.loc[list(known), "seed"].dropna()); nf_tr = set(fam.loc[list(known), "name_family"].dropna())
attr = pd.DataFrame(index=mir[new])
attr["alias"] = chk.identical_hairpin_to.notna() | chk.identical_mature_to.notna()
attr["identity"] = chk.max_identity_to_known
attr["seed_known"] = [isinstance(fam.loc[m, "seed"], str) and fam.loc[m, "seed"] in seeds_tr for m in attr.index]
attr["namefam_known"] = [fam.loc[m, "name_family"] in nf_tr for m in attr.index]
attr["identity_bin"] = pd.cut(attr.identity, [-1, 0.5, 0.8, 1.01], labels=["<0.5", "0.5-0.8", ">=0.8"], right=False)
attr.to_csv("results/prospective_new_mirna_attributes.csv")

def load(neg):
    runs = []
    for seed in range(10):
        d = {f: np.load(P / f"tcold_{neg}_mlp_{f}_degree_s{seed}.npz") for f in FEATS}
        pairs, y = d[FEATS[0]]["pairs"], d[FEATS[0]]["y"]
        sc = {f: d[f]["score"] for f in FEATS}
        sc["popularity"] = deg_d[pairs[:, 1]] + 1e-9 * np.random.default_rng(seed).random(len(y))
        runs.append((pairs, y, sc))
    return runs

ALL = FEATS + ["popularity"]
rng = np.random.default_rng(0); ci, strat = [], []
for neg in ["degree", "uniform"]:
    runs = load(neg)
    def evaluate(keep_m, boot=False, nb=NB):
        """mean-over-seeds AUROC for pairs whose miRNA is in keep_m; optionally miRNA bootstrap"""
        prep = []
        for pairs, y, sc in runs:
            mnames = mir[pairs[:, 0]]; idx = {m: np.flatnonzero(mnames == m) for m in keep_m}
            idx = {m: v for m, v in idx.items() if len(v)}
            ii = np.concatenate(list(idx.values())); prep.append((y, sc, idx, ii))
        point = {f: np.mean([auc(y[ii], sc[f][ii]) for y, sc, _, ii in prep]) for f in ALL}
        per_seed = {f: np.array([auc(y[ii], sc[f][ii]) for y, sc, _, ii in prep]) for f in ALL}
        if not boot: return point, per_seed, None
        ms = np.array(sorted(prep[0][2])); bs = {f: np.zeros(nb) for f in ALL}
        for b in range(nb):
            draw = rng.choice(ms, len(ms), replace=True); acc = {f: [] for f in ALL}
            for y, sc, idx, _ in prep:
                jj = np.concatenate([idx[m] for m in draw if m in idx]); yy = y[jj]
                if yy.min() == yy.max(): continue
                for f in ALL: acc[f].append(auc(yy, sc[f][jj]))
            for f in ALL: bs[f][b] = np.mean(acc[f])
        return point, per_seed, bs
    for label, keep in [("all", attr.index), ("no_alias", attr.index[~attr.alias])]:
        pt, ps, bs = evaluate(list(keep), boot=True)
        rows = []
        for a_, b_ in COMPS:
            d = bs[a_] - bs[b_]; lo, hi = np.percentile(d, [2.5, 97.5]); l9, h9 = np.percentile(d, [5, 95])
            pw = wilcoxon(ps[a_], ps[b_]).pvalue
            rows.append(dict(neg=neg, subset=label, n_mirna=len(keep), a=a_, b=b_, auc_a=pt[a_], auc_b=pt[b_],
                             diff=pt[a_] - pt[b_], ci_low=lo, ci_high=hi, ci90_low=l9, ci90_high=h9,
                             p_boot=2 * min((d <= 0).mean(), (d >= 0).mean()), p_wilcoxon=pw,
                             wins=int((ps[a_] > ps[b_]).sum()), equivalent_0_01=bool(l9 > -0.01 and h9 < 0.01)))
        r = pd.DataFrame(rows); order = r.p_boot.sort_values().index; run = 0; m = len(r)
        for k, i in enumerate(order): run = max(run, min(1, (m - k) * r.loc[i, "p_boot"])); r.loc[i, "p_holm"] = run
        ci.append(r)
    for col in ["identity_bin", "seed_known", "namefam_known"]:
        for lev, g in attr.groupby(col, observed=True):
            pt, ps, _ = evaluate(list(g.index))
            npairs = int(sum(np.isin(mir[runs[0][0][:, 0]], g.index) & (runs[0][1] == 1)))
            strat.append(dict(neg=neg, stratum=col, level=str(lev), n_mirna=len(g), n_pos=npairs,
                              **{f: pt[f] for f in ALL}, ernie_minus_random=pt["ernierna_both"] - pt["random"],
                              align_minus_random=pt["align"] - pt["random"]))
ci = pd.concat(ci); st = pd.DataFrame(strat)
ci.round(4).to_csv("results/prospective_bootstrap.csv", index=False); st.round(4).to_csv("results/prospective_strata.csv", index=False)
pd.set_option("display.width", 250)
print(ci.round(4).to_string()); print(st.round(4).to_string())
