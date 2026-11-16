"""fig5: gain over random miRNA features as test miRNAs become less related to training miRNAs
(popularity-matched negatives, MLP with popularity-debiased training, miRNA-level bootstrap 95% CI)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "savefig.dpi": 300})
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b"
S = [("Random\nmiRNA folds", "results/bootstrap_robust_kfold_diff_ci.csv", "kfold"),
     ("Seed\nfamilies", "results/bootstrap_robust_seed_family_diff_ci.csv", "seed_family"),
     ("Name\nfamilies", "results/bootstrap_robust_name_family_diff_ci.csv", "name_family"),
     ("Homology\nclusters", "results/bootstrap_robust_homology70_diff_ci.csv", "homology70")]
summ = pd.read_csv("results/robust_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [5, 4]})
for ax, neg, title in [(axes[0], "degree", "(a) Popularity-matched negatives"), (axes[1], "exact", "(b) Exact degree-preserving negatives")]:
    labels = []
    for i, (lab, f, sp) in enumerate(S + ([("Prospective\nHMDD v4.0", None, None)] if neg == "degree" else [])):
        labels.append(lab)
        if f is None:
            d = pd.read_csv("results/prospective_bootstrap.csv"); d = d[(d.neg == "degree") & (d.subset == "all")]
            pop = d[(d.a == 'ernierna_both') & (d.b == 'popularity')].auc_b.iloc[0] - d[(d.a == 'ernierna_both') & (d.b == 'random')].auc_b.iloc[0]
        else:
            d = pd.read_csv(f); d = d[d.neg == neg]
            e = summ[(summ.split == sp) & (summ.neg == neg)]
            pop = e[e.feature == "popularity"].auc.iloc[0] - e[(e.feature == "random") & (e.model == "MLP (debiased)")].auc.iloc[0]
        for a, col, dx, nm in [("ernierna_both", BLUE, -0.12, "ERNIE-RNA"), ("align", ORANGE, 0.12, "Alignment")]:
            r = d[(d.a == a) & (d.b == "random")].iloc[0]
            ax.errorbar(i + dx, r["diff"], yerr=[[r["diff"] - r.ci_low], [r.ci_high - r["diff"]]], fmt="o", ms=5.5,
                        color=col, lw=1.4, capsize=0, label=nm if i == 0 else None, zorder=3)
        ax.plot(i, pop, marker="D", ms=5, mfc="white", mec=INK, mew=1.1, ls="none", label="Popularity only" if i == 0 else None, zorder=3)
    ax.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.grid(axis="y", color="#e8e7e3", lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("AUROC gain over random miRNA features")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
fig.text(0.5, 0.0, "test miRNAs less related to training miRNAs →", ha="center", fontsize=7.5, color="#52514e")
fig.tight_layout()
for ext in ["pdf", "png"]: fig.savefig(f"figures/fig5_novelty_gradient.{ext}", bbox_inches="tight")
