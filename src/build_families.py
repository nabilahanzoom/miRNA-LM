"""miRNA groups for family-out validation.
name_family : miRBase-style family from the precursor name (mir-200a/b/c -> mir-200; let-7a..i -> let-7)
seed_family : mature-miRNA seed (nt 2-8), 5p arm preferred; miRNAs sharing a seed form one family
combined    : union of both (connected components), the strictest grouping
Writes data/processed/mirna_family.csv"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
RAW, PROC = Path(sys.argv[1]), Path(sys.argv[2])
def fasta(p):
    out, h, buf = [], None, []
    for l in open(p):
        l = l.strip()
        if l.startswith(">"):
            if h: out.append((h, "".join(buf)))
            h, buf = l[1:].split()[0].lower(), []
        else: buf.append(l)
    out.append((h, "".join(buf)))
    return [(n, s.upper().replace("T", "U")) for n, s in out if n.startswith("hsa-")]
mt = fasta(RAW / "mature.fa")
mir = pd.read_csv(PROC / "mirnas.csv").mirna.tolist()
rows = []
for m in mir:
    core = m.replace("hsa-", "")
    r = re.match(r"(let-7|lin-4|mir-\d+)", core)
    name_fam = r.group(1) if r else core
    pat = re.compile(re.escape(m) + r"(-\d+)?(-[35]p)?$")
    recs = [(n, s) for n, s in mt if pat.match(n)]
    five = [s for n, s in recs if n.endswith("-5p")] or [s for n, s in recs if not n.endswith("-3p")] or [s for _, s in recs]
    seed = five[0][1:8] if five else None
    rows.append(dict(mirna=m, name_family=name_fam, seed=seed))
g = pd.DataFrame(rows)
g["seed_family"] = [s if s else f"noseed_{i}" for i, s in enumerate(g.seed)]
# combined: connected components over shared name family OR shared seed
parent = list(range(len(g)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for col in ["name_family", "seed_family"]:
    first = {}
    for i, v in enumerate(g[col]):
        if v in first: parent[find(i)] = find(first[v])
        else: first[v] = i
g["combined_family"] = [f"c{find(i)}" for i in range(len(g))]
# homology clusters: connected components of alignment similarity >= 0.7 (mature, hairpin fallback)
from scipy.sparse.csgraph import connected_components
hm = np.load(PROC / "has_mature.npy")
S = np.where(hm[:, None] & hm[None, :], np.load(PROC / "seqsim_mature_max.npy"), np.load(PROC / "seqsim_hairpin_max.npy"))
np.fill_diagonal(S, 0)
g["homology70"] = [f"h{l}" for l in connected_components(S >= 0.7, directed=False)[1]]
g.to_csv(PROC / "mirna_family.csv", index=False)
for col in ["name_family", "seed_family", "combined_family", "homology70"]:
    s = g[col].value_counts()
    print(f"{col}: {len(s)} groups, singletons {int((s == 1).sum())}, largest {s.iloc[0]} ({s.index[0]}), miRNAs in multi-member groups {int(s[s > 1].sum())}")
