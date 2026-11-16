"""Supplementary tables for the robustness analyses -> tables/table_s3.tex ... table_s5.tex"""
import pandas as pd
from pathlib import Path
NM = {"ernierna_both": "ERNIE-RNA", "rnafm_both": "RNA-FM", "rinalmo_both": "RiNALMo-giga", "align": "Alignment",
      "kmer": "$k$-mer", "random": "Random", "popularity": "Popularity"}
SPL = {"kfold": "Random folds", "seed_family": "Seed families", "name_family": "Name families", "homology70": "Homology clusters"}
NEG = {"uniform": "Uniform", "degree": "Pop.-matched", "exact": "Exact"}
T = Path("tables"); T.mkdir(exist_ok=True)

# S3: all robust results
s = pd.read_csv("results/robust_summary.csv")
L = [r"\begin{longtable}{lllccc}", r"\caption{Cold-miRNA AUROC (mean $\pm$ s.d., 3 seeds $\times$ 5 folds) for every partitioning scheme, predictor and test-negative protocol. MLP: popularity-debiased training; IMC: uniform training negatives.}\label{tab:s3}\\",
     r"\toprule Split & Model & Representation & Uniform & Pop.-matched & Exact\\\midrule\endhead"]
for sp in SPL:
    for model in ["MLP (debiased)", "IMC", "none"]:
        g = s[(s.split == sp) & (s.model == model)]
        for f in NM:
            r = g[g.feature == f]
            if r.empty: continue
            cell = lambda n: (lambda x: f"{x.auc.iloc[0]:.3f} $\\pm$ {x.sd.iloc[0]:.3f}" if len(x) else "--")(r[r.neg == n])
            L.append(f"{SPL[sp]} & {'--' if model == 'none' else model.replace(' (debiased)', '')} & {NM[f]} & {cell('uniform')} & {cell('degree')} & {cell('exact')}\\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{longtable}")
(T / "table_s3.tex").write_text("\n".join(L) + "\n")

# S4: bootstrap CIs
src = [("Main CV", "results/bootstrap_cv_debiased_diff_ci.csv"), ("Random folds", "results/bootstrap_robust_kfold_diff_ci.csv"),
       ("Seed families", "results/bootstrap_robust_seed_family_diff_ci.csv"), ("Name families", "results/bootstrap_robust_name_family_diff_ci.csv"),
       ("Homology clusters", "results/bootstrap_robust_homology70_diff_ci.csv")]
L = [r"{\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{lllcccc}", r"\caption{Paired AUROC differences with miRNA-level bootstrap 95\% confidence intervals (1{,}000 replicates; MLP, popularity-debiased training). Equiv.: 90\% interval inside $\pm0.01$. $p_\mathrm{Holm}$: Wilcoxon over seed $\times$ fold (cross-validation) or bootstrap (prospective), Holm-adjusted within each block.}\label{tab:s4}\\",
     r"\toprule Split & Negatives & Comparison & $\Delta$AUROC & 95\% CI & Equiv. & $p_\mathrm{Holm}$\\\midrule\endhead"]
for lab, f in src:
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        if lab == "Main CV" and r.neg == "exact": continue
        L.append(f"{lab} & {NEG[r.neg]} & {NM[r.a]} -- {NM[r.b]} & {r['diff']:+.4f} & [{r.ci_low:+.4f}, {r.ci_high:+.4f}] & {'yes' if r.equivalent_0_01 else 'no'} & {('$<$0.001' if r.p_holm < 0.001 else f'{r.p_holm:.3f}')}\\\\")
    L.append(r"\midrule")
p = pd.read_csv("results/prospective_bootstrap.csv")
for (neg, sub), d in p.groupby(["neg", "subset"], sort=False):
    lab = "Prosp." + (" no alias" if sub == "no_alias" else "")
    for _, r in d.iterrows():
        L.append(f"{lab} & {NEG[neg]} & {NM[r.a]} -- {NM[r.b]} & {r['diff']:+.4f} & [{r.ci_low:+.4f}, {r.ci_high:+.4f}] & {'yes' if r.equivalent_0_01 else 'no'} & {('$<$0.001' if r.p_holm < 0.001 else f'{r.p_holm:.3f}')}\\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{longtable}}")
(T / "table_s4.tex").write_text("\n".join(L) + "\n")

# S5: false negatives, novelty, prospective strata
fn = pd.read_csv("results/false_negatives_cv_summary.csv"); nov = pd.read_csv("results/novelty_comparison.csv")
st = pd.read_csv("results/prospective_strata.csv"); st = st[st.neg == "degree"]
L = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{4pt}", r"\caption{Robustness of the evaluation. (a) Cold-miRNA test negatives (HMDD v3.0, MLP, debiased training) that are reported as associations in HMDD v4.0, and AUROC before and after removing them. (b) Relatedness of test miRNAs to training miRNAs in cross-validation and in the prospective test. (c) Prospective AUROC (popularity-matched negatives, MLP, debiased training, mean of 10 seeds) by nearest-homolog identity and by seed or name family present among training miRNAs.}\label{tab:s5}",
     r"\textbf{(a)}\\[2pt]\begin{tabular}{lcccc}\toprule Representation & Negatives & Later reported & AUROC & AUROC (removed)\\\midrule"]
for _, r in fn[fn.feature.isin(["ernierna_both", "rnafm_both", "rinalmo_both", "align", "kmer", "random"])].iterrows():
    L.append(f"{NM[r.feature]} & {NEG[r.neg]} & {100*r.frac_fn:.1f}\\% & {r.auc:.3f} & {r.auc_clean:.3f}\\\\")
L += [r"\bottomrule\end{tabular}\\[8pt]", r"\textbf{(b)}\\[2pt]\begin{tabular}{lccccc}\toprule Test miRNAs & Median NN id. & NN $\ge0.8$ & NN $<0.5$ & Seed in train & Name fam.\ in train\\\midrule"]
for _, r in nov.iterrows():
    L.append(f"{ {"cross-validation (cold-miRNA)": "Cross-validation", "prospective (new in HMDD v4.0)": "Prospective (v4.0)"}.get(r.test, r.test)} & {r.nn_median:.3f} & {100*r.frac_ge_0_8:.1f}\\% & {100*r.frac_lt_0_5:.1f}\\% & {100*r.seed_in_train:.1f}\\% & {100*r.namefam_in_train:.1f}\\%\\\\")
L += [r"\bottomrule\end{tabular}\\[8pt]", r"\textbf{(c)}\\[2pt]\begin{tabular}{llcccccc}\toprule Stratum & Level & miRNAs & Positives & ERNIE-RNA & Alignment & Random & Popularity\\\midrule"]
LV = {"<0.5": "$<0.5$", "0.5-0.8": "0.5--0.8", ">=0.8": "$\\ge0.8$", "True": "yes", "False": "no"}
SN = {"identity_bin": "NN identity", "seed_known": "Seed in training", "namefam_known": "Name family in training"}
for _, r in st.iterrows():
    L.append(f"{SN[r.stratum]} & {LV.get(r["level"], r["level"])} & {r.n_mirna} & {r.n_pos} & {r.ernierna_both:.3f} & {r["align"]:.3f} & {r["random"]:.3f} & {r["popularity"]:.3f}\\\\")
L += [r"\bottomrule\end{tabular}", r"\end{table}"]
(T / "table_s5.tex").write_text("\n".join(L) + "\n")
print("ok")
