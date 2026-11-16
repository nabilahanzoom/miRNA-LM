"""How often is a miRNA-disease association reported in the same paper as the same disease for a close
sequence homolog?  (A mechanism by which random cold-miRNA CV can reward sequence similarity.)"""
import numpy as np, pandas as pd, sys
RAW = sys.argv[1]; P = "data/processed/"
h = pd.read_csv(f"{RAW}/hmdd_associations.txt", sep="\t", dtype=str, encoding="latin1")
h["mir"] = h.mir.str.strip().str.lower(); h["disease"] = h.disease.str.strip()
mir = pd.read_csv(P + "mirnas.csv").mirna.tolist(); dis = pd.read_csv(P + "diseases.csv").disease.tolist()
mi = {m: i for i, m in enumerate(mir)}; di = {d: j for j, d in enumerate(dis)}
h = h[h.mir.isin(mi) & h.disease.isin(di)]
pm = h.groupby(["mir", "disease"]).pmid.apply(set).to_dict()
S = np.where(np.load(P + "has_mature.npy")[:, None] & np.load(P + "has_mature.npy")[None, :],
             np.load(P + "seqsim_mature_max.npy"), np.load(P + "seqsim_hairpin_max.npy"))
np.fill_diagonal(S, 0)
by_d = {}
for (m, d) in pm: by_d.setdefault(d, []).append(m)
rows = []
for thr in [0.5, 0.8]:
    same_paper = any_assoc = 0; n = 0
    for (m, d), ps in pm.items():
        n += 1; i = mi[m]
        hom = [o for o in by_d[d] if o != m and S[i, mi[o]] >= thr]
        any_assoc += bool(hom)
        same_paper += any(ps & pm[(o, d)] for o in hom)
    rows.append(dict(homolog_threshold=thr, pairs=n, homolog_has_same_disease=any_assoc / n, homolog_same_disease_same_paper=same_paper / n))
r = pd.DataFrame(rows).round(3); r.to_csv("results/coreporting.csv", index=False); print(r.to_string(index=False))
