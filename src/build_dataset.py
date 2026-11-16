"""Build the miRNA-disease dataset.

Outputs (data/processed/):
  mirnas.csv            HMDD miRNA names kept (mapped to >=1 RNA-FM embedding)
  diseases.csv          HMDD disease names with MeSH descriptor IDs (if any)
  assoc.npy             (n_mirna, n_disease) binary association matrix
  emb_mature.npy        RNA-FM mature embeddings (mean of mapped mature arms; 0 if none)
  emb_hairpin.npy       RNA-FM hairpin embeddings (mean of mapped precursors; 0 if none)
  has_mature.npy, has_hairpin.npy  availability masks
  dis_sem_sim.npy       MeSH semantic similarity (Wang et al. 2007, delta=0.5)
  mapping_report.txt    coverage statistics
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1])            # folder with HMDD, MeSH xml
EMB = Path(sys.argv[2])            # folder with rnafm_* files
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
report = []

# ---------------------------------------------------------------- HMDD
h = pd.read_csv(RAW / "hmdd_associations.txt", sep="\t", dtype=str, encoding="latin1")
h["mir"] = h["mir"].str.strip().str.lower()
h["disease"] = h["disease"].str.strip()
pairs = h[["mir", "disease"]].drop_duplicates()
report.append(f"HMDD records: {len(h)}; unique pairs: {len(pairs)}; "
              f"miRNAs: {pairs.mir.nunique()}; diseases: {pairs.disease.nunique()}")

# ---------------------------------------------------------------- miRNA -> embeddings
hp_names = pd.read_csv(EMB / "rnafm_hairpin_names.csv").name.tolist()
mt_names = pd.read_csv(EMB / "rnafm_mature_names.csv").name.tolist()
E_hp = np.load(EMB / "rnafm_hairpin_embeddings.npy")
E_mt = np.load(EMB / "rnafm_mature_embeddings.npy")
hp_idx = {n.lower(): i for i, n in enumerate(hp_names)}
mt_low = [n.lower() for n in mt_names]


def map_hairpin(m):
    if m in hp_idx:
        return [hp_idx[m]]
    pat = re.compile(re.escape(m) + r"(-\d+)?$")          # hsa-let-7a -> hsa-let-7a-1/2/3
    return [i for k, i in hp_idx.items() if pat.match(k)]


def map_mature(m):
    pat = re.compile(re.escape(m) + r"(-\d+)?(-[35]p)?$")  # hsa-mir-21 -> hsa-mir-21-5p/3p
    return [i for i, k in enumerate(mt_low) if pat.match(k)]


rows = []
for m in sorted(pairs.mir.unique()):
    a, b = map_hairpin(m), map_mature(m)
    if a or b:
        rows.append((m, a, b))
mirnas = [r[0] for r in rows]
d = E_hp.shape[1]
emb_hp = np.zeros((len(rows), d), np.float32)
emb_mt = np.zeros((len(rows), d), np.float32)
has_hp = np.zeros(len(rows), bool)
has_mt = np.zeros(len(rows), bool)
for i, (_, a, b) in enumerate(rows):
    if a:
        emb_hp[i] = E_hp[a].mean(0); has_hp[i] = True
    if b:
        emb_mt[i] = E_mt[b].mean(0); has_mt[i] = True
unmapped = sorted(set(pairs.mir) - set(mirnas))
report.append(f"miRNAs mapped: {len(mirnas)} (hairpin {has_hp.sum()}, mature {has_mt.sum()}); "
              f"unmapped: {len(unmapped)}")
report.append("Unmapped miRNAs: " + ", ".join(unmapped))

pairs = pairs[pairs.mir.isin(mirnas)]
diseases = sorted(pairs.disease.unique())
mi = {m: i for i, m in enumerate(mirnas)}
di = {x: j for j, x in enumerate(diseases)}
A = np.zeros((len(mirnas), len(diseases)), np.int8)
for m, x in pairs.itertuples(index=False):
    A[mi[m], di[x]] = 1
report.append(f"Final: {len(mirnas)} miRNAs x {len(diseases)} diseases, {int(A.sum())} associations, "
              f"density {A.mean():.4f}")

# ---------------------------------------------------------------- MeSH
print("Parsing MeSH ...", flush=True)
ui2trees, ui2name, term2ui = {}, {}, {}
for _, el in ET.iterparse(RAW / "desc2026.xml", events=("end",)):
    if el.tag != "DescriptorRecord":
        continue
    ui = el.findtext("DescriptorUI")
    name = el.findtext("DescriptorName/String")
    trees = [t.text for t in el.findall("TreeNumberList/TreeNumber")]
    ui2trees[ui] = trees
    ui2name[ui] = name
    for t in el.iter("Term"):
        s = t.findtext("String")
        if s:
            term2ui.setdefault(s.lower(), ui)
    el.clear()
tree2ui = {t: u for u, ts in ui2trees.items() for t in ts}
print(f"MeSH descriptors: {len(ui2trees)}", flush=True)

# disease -> MeSH UI: HMDD mesh column first, else exact name/entry-term match
hm = h[["disease", "mesh"]].dropna().drop_duplicates("disease").set_index("disease").mesh.to_dict()
dis_ui, how = [], []
for x in diseases:
    u = hm.get(x)
    if u is not None:
        u = u.split(";")[0].strip()
    if u in ui2trees:
        dis_ui.append(u); how.append("hmdd_mesh")
    elif x.lower() in term2ui:
        dis_ui.append(term2ui[x.lower()]); how.append("name_match")
    else:
        dis_ui.append(None); how.append("none")
how = pd.Series(how)
report.append("Disease MeSH mapping: " + how.value_counts().to_dict().__str__())


def dag_values(ui, delta=0.5):
    """Wang semantic contribution of every ancestor of descriptor ui."""
    vals = {ui: 1.0}
    for t in ui2trees.get(ui, []):
        parts = t.split(".")
        for k in range(len(parts) - 1, 0, -1):
            anc = tree2ui.get(".".join(parts[:k]))
            if anc is None:
                continue
            v = delta ** (len(parts) - k)
            if v > vals.get(anc, 0):
                vals[anc] = v
    return vals


dv = [dag_values(u) if u else None for u in dis_ui]
n = len(diseases)
S = np.eye(n, dtype=np.float32)
for i in range(n):
    if dv[i] is None:
        continue
    si = sum(dv[i].values())
    for j in range(i + 1, n):
        if dv[j] is None:
            continue
        common = dv[i].keys() & dv[j].keys()
        if common:
            s = sum(dv[i][t] + dv[j][t] for t in common) / (si + sum(dv[j].values()))
            S[i, j] = S[j, i] = s
report.append(f"Disease semantic similarity: mean off-diag {S[~np.eye(n, dtype=bool)].mean():.4f}")

# ---------------------------------------------------------------- save
pd.DataFrame({"mirna": mirnas, "has_hairpin": has_hp, "has_mature": has_mt}).to_csv(OUT / "mirnas.csv", index=False)
pd.DataFrame({"disease": diseases, "mesh_ui": dis_ui, "mesh_map": how}).to_csv(OUT / "diseases.csv", index=False)
np.save(OUT / "assoc.npy", A)
np.save(OUT / "emb_hairpin.npy", emb_hp)
np.save(OUT / "emb_mature.npy", emb_mt)
np.save(OUT / "has_hairpin.npy", has_hp)
np.save(OUT / "has_mature.npy", has_mt)
np.save(OUT / "dis_sem_sim.npy", S)
(OUT / "mapping_report.txt").write_text("\n".join(report) + "\n")
print("\n".join(report[:3] + report[4:]))
