"""Label-free / degree-only baselines on the exact same splits as experiment.py.

popularity : score(m, d) = training degree of d (+ training degree of m, random split only)
Also reports per-entity AUC (ranking diseases within each held-out miRNA) which removes
miRNA-degree effects, computed from saved prediction files if given.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from experiment import load, make_splits  # noqa: E402

D = load(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
A = D["A"]
rows = []
for seed in [0, 1, 2]:
  for setting in ["random", "cold_mirna", "cold_disease"]:
    for fold, (region, tp, tns, cold) in enumerate(make_splits(A, setting, seed)):
      for mode, tn in tns.items():
        A_tr = A * region
        dm, dd = A_tr.sum(1), A_tr.sum(0)
        te = np.concatenate([tp, tn]); y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]
        if setting == "random":
            s = np.log1p(dm[te[:, 0]]) + np.log1p(dd[te[:, 1]])
        elif setting == "cold_mirna":
            s = dd[te[:, 1]]
        else:
            s = dm[te[:, 0]]
        s = s + 1e-9 * np.random.default_rng(fold).random(len(s))
        rows.append(dict(setting=setting, neg=mode, seed=seed, fold=fold, auc=roc_auc_score(y, s),
                         aupr=average_precision_score(y, s)))
df = pd.DataFrame(rows)
df.to_csv("results/popularity_baseline.csv", index=False)
print(df.groupby(["setting", "neg"])[["auc", "aupr"]].agg(["mean", "std"]).round(4))
