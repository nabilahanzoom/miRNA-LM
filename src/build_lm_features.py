"""Map embeddings of every RNA language model onto data/processed/mirnas.csv.

Writes data/processed/emb_<model>_<mature|hairpin>.npy using the same name-mapping
rules as build_dataset.py (mean over mapped records; zeros if none). Checks that the
RNA-FM re-extraction matches the original features used in build_dataset.py.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EMB, PROC = Path(sys.argv[1]), Path(sys.argv[2])
mir = pd.read_csv(PROC / "mirnas.csv")

for model in ["rnafm", "ernierna", "rinalmo"]:
    for kind in ["mature", "hairpin"]:
        names = pd.read_csv(EMB / f"{model}_{kind}_names.csv").name.str.lower().tolist()
        E = np.load(EMB / f"{model}_{kind}_embeddings.npy")
        idx = {n: i for i, n in enumerate(names)}
        X = np.zeros((len(mir), E.shape[1]), np.float32)
        for r, m in enumerate(mir.mirna):
            if kind == "hairpin":
                if m in idx:
                    a = [idx[m]]
                else:
                    pat = re.compile(re.escape(m) + r"(-\d+)?$")
                    a = [i for k, i in idx.items() if pat.match(k)]
            else:
                pat = re.compile(re.escape(m) + r"(-\d+)?(-[35]p)?$")
                a = [i for i, k in enumerate(names) if pat.match(k)]
            assert bool(a) == bool(mir[f"has_{kind}"][r]), (model, kind, m)
            if a:
                X[r] = E[a].mean(0)
        np.save(PROC / f"emb_{model}_{kind}.npy", X)
        print(f"{model:9s} {kind:8s} {X.shape}")

for kind in ["mature", "hairpin"]:
    d = np.abs(np.load(PROC / f"emb_rnafm_{kind}.npy") - np.load(PROC / f"emb_{kind}.npy")).max()
    print(f"RNA-FM {kind}: max |re-extracted - original| = {d:.4f}")
