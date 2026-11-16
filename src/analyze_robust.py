"""Summarise robustness runs (exact degree-preserving negatives, family-out splits).
Writes results/robust_summary.csv (mean, sd per split x model x negative x feature, incl. popularity baseline)
and tables/table_robust.tex.   python src/analyze_robust.py"""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

A = np.load("data/processed/assoc.npy")
SPLITS = {"kfold": ("robust_kfold_degree", "robust_kfold_imc", "preds_robust"),
          "homology70": ("robust_homology70", "robust_homology70_imc", "preds_robust_hom70"),
          "name_family": ("robust_name_family", "robust_name_family_imc", "preds_robust_namefam"),
          "seed_family": ("robust_seed_family", "robust_seed_family_imc", "preds_robust_seedfam")}
rows = []
for split, (mlp, imc, pdir) in SPLITS.items():
    for tag, f in [("MLP (debiased)", mlp), ("IMC", imc)]:
        p = Path(f"results/{f}.csv")
        if not p.exists(): continue
        d = pd.read_csv(p).drop_duplicates(["model", "feature", "seed", "fold", "neg"])
        for (neg, feat), g in d.groupby(["neg", "feature"]):
            rows.append(dict(split=split, model=tag, neg=neg, feature=feat, auc=g.auc.mean(), sd=g.auc.std(), n=len(g),
                             dropped=g.exact_dropped.mean() if neg == "exact" else 0))
    # popularity baseline from saved prediction pairs (training disease degree)
    for neg in ["uniform", "degree", "exact"]:
        aucs = []
        for fn in glob.glob(f"results/{pdir}/*_{neg}_mlp_random_degree_s*_f*.npz"):
            z = np.load(fn); pairs, y = z["pairs"], z["y"]
            held = np.unique(pairs[:, 0]); A_tr = A.copy(); A_tr[held] = 0
            aucs.append(roc_auc_score(y, A_tr.sum(0)[pairs[:, 1]]))
        if aucs:
            rows.append(dict(split=split, model="none", neg=neg, feature="popularity", auc=np.mean(aucs), sd=np.std(aucs, ddof=1), n=len(aucs), dropped=0))
import os, sys
if "--table_only" in sys.argv: s = pd.read_csv("results/robust_summary.csv")
else: s = pd.DataFrame(rows); s.round(6).to_csv("results/robust_summary.csv", index=False)
pd.set_option("display.width", 200)
print(s.pivot_table(index=["split", "model", "feature"], columns="neg", values="auc").round(4).to_string())

# LaTeX table: AUROC under exact degree-preserving negatives, MLP debiased and IMC, per split
NAMES = {"ernierna_both": "ERNIE-RNA", "rnafm_both": "RNA-FM", "rinalmo_both": "RiNALMo-giga", "align": "Alignment",
         "kmer": "$k$-mer", "random": "Random features", "popularity": "Popularity only"}
SPL = {"kfold": "Random miRNA folds", "seed_family": "Seed families", "name_family": "Name families", "homology70": "Homology clusters"}
cols = [(sp, m) for sp in SPL for m in ["MLP (debiased)", "IMC"]]
lines = [r"\begin{table}[t]\centering\small", r"\caption{Cold-miRNA AUROC with exact degree-preserving test negatives under four partitioning schemes. "
         r"Each value is the mean over 3 seeds $\times$ 5 folds. Popularity only ranks diseases by training degree; under exact negatives it is close to chance by construction. "
         r"Bold: best per column. --: not run.}\label{tab:robust}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "cc" * len(SPL) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{v}}}" for v in SPL.values()) + r"\\",
         " ".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(SPL))),
         "Representation & " + " & ".join(["MLP & IMC"] * len(SPL)) + r"\\", r"\midrule"]
e = s[s.neg == "exact"]
best = {c: e[(e.split == c[0]) & (e.model == c[1])].auc.max() for c in cols}
for f, nm in NAMES.items():
    cells = []
    for sp, m in cols:
        mm = "none" if f == "popularity" else m
        v = e[(e.split == sp) & (e.model == mm) & (e.feature == f)].auc
        if f == "popularity" and m == "IMC": cells.append(""); continue
        if v.empty: cells.append("--"); continue
        t = f"{v.iloc[0]:.3f}"; cells.append(rf"\textbf{{{t}}}" if abs(v.iloc[0] - best[(sp, m)]) < 1e-9 else t)
    if f == "popularity": lines.append(r"\midrule")
    lines.append(nm + " & " + " & ".join(cells) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
Path("tables").mkdir(exist_ok=True); Path("tables/table_robust.tex").write_text("\n".join(lines) + "\n")
