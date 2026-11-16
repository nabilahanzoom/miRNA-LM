"""Earliest-evidence PMID for every miRNA-disease pair (PMIDs are assigned ~chronologically).
Writes data/processed/first_pmid.npy (int64, 0 = no association), aligned with assoc.npy."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
RAW, PROC = Path(sys.argv[1]), Path(sys.argv[2])
h = pd.read_csv(RAW / "hmdd_associations.txt", sep="\t", dtype=str, encoding="latin1")
h["mir"] = h["mir"].str.strip().str.lower(); h["disease"] = h["disease"].str.strip()
h["pmid"] = pd.to_numeric(h.pmid, errors="coerce")
print("records without numeric PMID:", h.pmid.isna().sum())
first = h.dropna(subset=["pmid"]).groupby(["mir", "disease"]).pmid.min()
mir = pd.read_csv(PROC / "mirnas.csv").mirna.tolist(); dis = pd.read_csv(PROC / "diseases.csv").disease.tolist()
mi = {m: i for i, m in enumerate(mir)}; di = {d: j for j, d in enumerate(dis)}
A = np.load(PROC / "assoc.npy"); T = np.zeros(A.shape, np.int64)
for (m, d), p in first.items():
    if m in mi and d in di: T[mi[m], di[d]] = int(p)
assert ((T > 0) == (A > 0)).all(), ((T > 0) != (A > 0)).sum()
np.save(PROC / "first_pmid.npy", T)
fm = np.where(A.any(1), np.where(T > 0, T, np.iinfo(np.int64).max).min(1), 0)
for c in [26_000_000, 28_000_000, 30_000_000, 31_000_000, 32_000_000, 33_000_000, 34_000_000]:
    new = (fm > c).sum(); print(f"cutoff PMID {c:,}: train pairs {(T[(T>0)]<=c).sum()}, new miRNAs {new}, "
          f"their pairs {(A[fm > c]).sum()}, new pairs of old miRNAs {((T>c)&(fm<=c)[:,None]).sum()}")
