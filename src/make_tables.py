"""LaTeX tables for the manuscript."""
import pandas as pd, numpy as np
N = {"ernierna_both+align": "ERNIE-RNA + align", "rnafm_both+align": "RNA-FM + align", "ernierna_both": "ERNIE-RNA",
     "rnafm_both": "RNA-FM", "rinalmo_both": "RiNALMo-giga", "align": "SW alignment", "kmer": "$k$-mer",
     "random": "Random features", "popularity": "Popularity only", "sim": "FS + GIP (classical)"}
s = pd.read_csv("results/summary_all.csv"); s = s[s.setting == "cold_mirna"]
order = ["ernierna_both", "rnafm_both", "rinalmo_both", "align", "kmer", "ernierna_both+align", "rnafm_both+align", "random", "popularity"]
cfg = [("uniform", "uniform"), ("degree", "uniform"), ("degree", "degree")]
L = [r"\begin{table}[t]\centering\small", r"\caption{Cold-miRNA five-fold cross-validation (MLP; mean $\pm$ s.d. AUROC over 3 seeds $\times$ 5 folds; AUPR in parentheses). The popularity baseline ranks diseases by their training degree.}\label{tab:cv}",
     r"\resizebox{\linewidth}{!}{\begin{tabular}{lccc}\toprule", r" & Uniform negatives & \multicolumn{2}{c}{Popularity-matched negatives}\\\cmidrule(lr){3-4}",
     r"Representation & standard training & standard training & debiased training\\\midrule"]
for f in order:
    cells = []
    for neg, tn in cfg:
        q = s[(s.feature == f) & (s.neg == neg) & ((s.train_neg == tn) | (f == "popularity")) & s.model.isin(["mlp", "none"])]
        q = q.iloc[0]; cells.append(f"{q.auc_mean:.3f} $\\pm$ {q.auc_std:.3f} ({q.aupr_mean:.3f})")
    if f == "random": L.append(r"\midrule")
    L.append(N[f] + " & " + " & ".join(cells) + r"\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open("results/table_cv.tex", "w").write("\n".join(L))

t = pd.read_csv("results/temporal_summary.csv")
L = [r"\begin{table}[t]\centering\small",
     r"\caption{Temporal validation, test set of emerging miRNAs (miRNAs whose first disease association appears after the training cut-off). AUROC, mean $\pm$ s.d. over seeds. Within-v3.0: cut-off mid-2014, 222 miRNAs. Prospective: trained on HMDD v3.0, tested on 443 miRNAs new in HMDD v4.0.}\label{tab:temporal}",
     r"\resizebox{\linewidth}{!}{\begin{tabular}{llcccc}\toprule",
     r" & & \multicolumn{2}{c}{Within HMDD v3.0} & \multicolumn{2}{c}{Prospective v3.0$\rightarrow$v4.0}\\\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
     r"Model & Representation & uniform neg. & pop.-matched neg. & uniform neg. & pop.-matched neg.\\\midrule"]
feats = ["ernierna_both", "rnafm_both", "rinalmo_both", "align", "kmer", "ernierna_both+align", "random"]
for mo, tn, lab in [("mlp", "degree", "MLP (debiased)"), ("mlp", "uniform", "MLP (standard)"), ("imc", "uniform", "IMC")]:
    for i, f in enumerate(feats):
        cells = []
        for sp in ["within_v3", "prospective"]:
            for neg in ["tcold_uniform", "tcold_degree"]:
                q = t[(t.split == sp) & (t.model == mo) & (t.train_neg == tn) & (t.feature == f) & (t.neg == neg)]
                cells.append(f"{q.auc_mean.iloc[0]:.3f} $\\pm$ {q.auc_std.iloc[0]:.3f}" if len(q) else "--")
        L.append((lab if i == 0 else "") + " & " + N[f] + " & " + " & ".join(cells) + r"\\")
    L.append(r"\midrule")
cells = []
for sp in ["within_v3", "prospective"]:
    for neg in ["tcold_uniform", "tcold_degree"]:
        q = t[(t.split == sp) & (t.feature == "popularity") & (t.neg == neg)]
        cells.append(f"{q.auc_mean.iloc[0]:.3f} $\\pm$ {q.auc_std.iloc[0]:.3f}")
L.append(r"-- & Popularity only & " + " & ".join(cells) + r"\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open("results/table_temporal.tex", "w").write("\n".join(L))

c = pd.read_csv("results/popularity_confounders.csv").set_index("feature")
p = pd.read_csv("results/covariate_probe.csv").set_index("feature")
L = [r"\begin{table}[t]\centering\small",
     r"\caption{What miRNA representations encode. Spearman $\rho$ of 5-fold cross-validated ridge predictions with miRNA research popularity ($\log$(1+degree)), with popularity after regressing out miRBase covariates (discovery order, conservation, length, availability), and with discovery order and conservation themselves. Covariates alone reach $\rho=" + f"{c.loc['covariates only','rho']:.2f}" + r"$ for popularity.}\label{tab:probe}",
     r"\begin{tabular}{lcccc}\toprule",
     r"Representation & Popularity & Popularity residual & Discovery order & Conservation\\\midrule"]
for f in ["ernierna_both", "rnafm_both", "rinalmo_both", "align", "kmer", "random"]:
    L.append(f"{N[f]} & {c.loc[f,'rho']:.2f} & {c.loc[f,'rho_residual']:.2f} & {p.loc[f].iloc[0]:.2f} & {p.loc[f].iloc[1]:.2f}" + r"\\")
L += [r"\bottomrule\end{tabular}\end{table}"]
open("results/table_probe.tex", "w").write("\n".join(L))
print("tables ok")

# Supplementary table S1: all settings, MLP and GNN, seed 0 features + IMC
s = pd.read_csv("results/summary_all.csv")
N2 = dict(N, rnafm_mature="RNA-FM (mature)", rnafm_hairpin="RNA-FM (hairpin)", **{"rnafm_both+sim": "RNA-FM + FS/GIP"})
L = [r"\begin{longtable}{lllcc}", r"\caption{All cross-validated results (AUROC mean $\pm$ s.d.; standard training unless noted).}\label{tab:s1}\\\toprule",
     r"Setting & Model & Representation & Uniform neg. & Pop.-matched neg.\\\midrule\endhead"]
for st in ["random", "cold_mirna", "cold_disease"]:
    for mo, tn in [("mlp", "uniform"), ("gnn", "uniform"), ("imc", "uniform"), ("mlp", "degree")]:
        x = s[(s.setting == st) & (s.model == mo) & (s.train_neg == tn)]
        if not len(x): continue
        for f in x.feature.unique():
            u = x[(x.feature == f) & (x.neg == "uniform")].iloc[0]; d = x[(x.feature == f) & (x.neg == "degree")].iloc[0]
            lab = {"mlp": "MLP", "gnn": "GNN", "imc": "IMC"}[mo] + (" (debiased)" if tn == "degree" else "")
            L.append(f"{st.replace('_','-')} & {lab} & {N2.get(f,f)} & {u.auc_mean:.3f} $\\pm$ {u.auc_std:.3f} & {d.auc_mean:.3f} $\\pm$ {d.auc_std:.3f}" + r"\\")
    p = s[(s.setting == st) & (s.feature == "popularity")]
    L.append(f"{st.replace('_','-')} & -- & Popularity only & {p[p.neg=='uniform'].auc_mean.iloc[0]:.3f} & {p[p.neg=='degree'].auc_mean.iloc[0]:.3f}" + r"\\\midrule")
L += [r"\bottomrule\end{longtable}"]
open("results/table_s1.tex", "w").write("\n".join(L))
print("S1 ok")
