import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "savefig.dpi": 300})
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b"
NAME = {"ernierna_both+align": "ERNIE-RNA + align", "rnafm_both+align": "RNA-FM + align",
        "ernierna_both": "ERNIE-RNA", "rnafm_both": "RNA-FM", "rinalmo_both": "RiNALMo",
        "align": "SW alignment", "kmer": "k-mer", "random": "Random features", "popularity": "Popularity only"}
GRP = lambda f: BLUE if (f.split("_")[0] in ("rnafm","ernierna","rinalmo") and "+" not in f) else (ORANGE if f in ("align", "kmer") else
                                                             ("#6250d6" if "+" in f else GRAY))
import os; os.makedirs("figures", exist_ok=True)
s = pd.read_csv("results/summary_all.csv")
s = s[s.setting == "cold_mirna"]
pop = s[s.feature == "popularity"].set_index("neg")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
order = ["ernierna_both+align", "rnafm_both+align", "ernierna_both", "rnafm_both", "rinalmo_both",
         "align", "kmer", "random"]
for ax, (neg, tn, title) in zip(axes, [("uniform", "uniform", "(a) Standard: uniform negatives"),
                                       ("degree", "degree", "(b) Popularity-matched negatives\n(debiased training)")]):
    d = s[(s.neg == neg) & (s.train_neg == tn) & (s.model == "mlp")].set_index("feature")
    for i, f in enumerate(order):
        ax.errorbar(d.auc_mean[f], i, xerr=d.auc_std[f], fmt="o", ms=6, color=GRP(f), ecolor=GRP(f), lw=1.5, capsize=0)
        ax.text(d.auc_mean[f] + d.auc_std[f] + 0.004, i, f"{d.auc_mean[f]:.3f}", va="center", fontsize=7.5, color="#52514e")
    p = pop.auc_mean[neg]
    ax.vlines(p, -0.5, len(order) - 0.5, color=INK, ls="--", lw=1)
    ax.text(p, -0.75, f"popularity only\n{p:.3f}", fontsize=7, color=INK, ha="center", va="bottom")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("AUROC (mean ± SD, 3 seeds × 5 folds)")
    ax.grid(axis="x", color="#e8e7e3", lw=0.6); ax.set_axisbelow(True)
axes[0].set_yticks(range(len(order)), [NAME[f] for f in order]); axes[0].set_ylim(len(order)-0.5, -1.5)
axes[0].set_xlim(0.76, 0.94); axes[1].set_xlim(0.57, 0.71)
fig.tight_layout(); fig.savefig("figures/fig1_cold_mirna_auc.png"); fig.savefig("figures/fig1_cold_mirna_auc.pdf")

# fig2: popularity encoding
e = pd.read_csv("results/popularity_encoding.csv").set_index("feature")
o = ["ernierna_both", "rnafm_both", "rinalmo_both", "align", "kmer", "random"]
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for i, f in enumerate(o):
    ax.barh(i, e.spearman[f], color=GRP(f), height=0.6)
    ax.text(max(e.spearman[f], 0) + 0.01, i, f"{e.spearman[f]:.2f}", va="center", fontsize=7.5, color="#52514e")
ax.set_yticks(range(len(o)), [NAME[f] for f in o]); ax.invert_yaxis()
ax.set_xlabel("Spearman ρ, predicted vs. true\nlog miRNA degree (5-fold ridge)"); ax.set_xlim(-0.05, 0.8)
ax.axvline(0, color=GRAY, lw=0.8)
fig.tight_layout(); fig.savefig("figures/fig2_popularity_encoding.png"); fig.savefig("figures/fig2_popularity_encoding.pdf")

# fig3: homology strata, degree eval
h = pd.read_csv("results/homology_strata_raw.csv")
h = h[(h.neg == "degree") & h.stratum.isin(["mid (0.5-0.8)", "high (>=0.8)"])]
g = h.groupby(["stratum", "feature"]).auc.agg(["mean", "std"])
o = ["ernierna_both", "rnafm_both", "rinalmo_both", "align", "kmer", "random"]
fig, ax = plt.subplots(figsize=(4.6, 2.8))
for k, (st, lab, mk) in enumerate([("mid (0.5-0.8)", "Nearest training homolog 0.5–0.8", "o"),
                                   ("high (>=0.8)", "Nearest training homolog ≥ 0.8", "s")]):
    for i, f in enumerate(o):
        y = i + (k - 0.5) * 0.3
        ax.errorbar(g.loc[(st, f), "mean"], y, xerr=g.loc[(st, f), "std"], fmt=mk, ms=5, color=GRP(f),
                    mfc=GRP(f) if k else "white", lw=1.2, label=lab if i == 0 else None)
ax.set_yticks(range(len(o)), [NAME[f] for f in o]); ax.invert_yaxis()
ax.set_xlabel("AUROC, popularity-matched negatives")
leg = ax.legend(fontsize=7, frameon=False, loc="lower right")
for hdl in leg.legend_handles: hdl.set_color(GRAY)
ax.grid(axis="x", color="#e8e7e3", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/fig3_homology_strata.png"); fig.savefig("figures/fig3_homology_strata.pdf")
print("ok")

# fig4: gain over random features, cross-validated cold-miRNA vs prospective emerging miRNAs (popularity-matched)
raw = pd.concat([pd.read_csv(f) for f in __import__("glob").glob("results/raw*.csv")]).drop_duplicates(
    ["setting", "model", "feature", "seed", "fold", "train_neg", "neg"])
pr = pd.concat([pd.read_csv(f"results/{f}.csv") for f in ["prosp_degree", "prosp_uniform"]])
feats = ["ernierna_both", "rnafm_both", "rinalmo_both", "align", "kmer", "ernierna_both+align"]
cfgs = [("mlp", "degree", "MLP, debiased training"), ("mlp", "uniform", "MLP, standard training"), ("imc", "uniform", "IMC")]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True, sharex=True)
for ax, (mo, tn, title) in zip(axes, cfgs):
    cv = raw[(raw.setting == "cold_mirna") & (raw.model == mo) & (raw.train_neg == tn) & (raw.neg == "degree")]
    cvp = cv.pivot_table(index=["seed", "fold"], columns="feature", values="auc")
    pp = pr[(pr.model == mo) & (pr.train_neg == tn) & (pr.neg == "tcold_degree")].pivot_table(index="seed", columns="feature", values="auc")
    for i, f in enumerate(feats):
        for k, (tab, mk, lab) in enumerate([(cvp, "o", "Cross-validated cold-miRNA"), (pp, "s", "Prospective: new miRNAs in HMDD v4.0")]):
            if f not in tab or "random" not in tab: continue
            d = (tab[f] - tab["random"]).dropna()
            y = i + (k - 0.5) * 0.32
            ax.errorbar(d.mean(), y, xerr=d.std(), fmt=mk, ms=5, color=GRP(f), mfc=GRP(f) if k else "white", lw=1.2,
                        label=lab if i == 0 else None)
    ax.axvline(0, color=INK, lw=0.8, ls="--")
    ax.set_title(title, fontsize=8.5, loc="left"); ax.grid(axis="x", color="#e8e7e3", lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("ΔAUROC vs. random features")
axes[0].set_yticks(range(len(feats)), [NAME[f] for f in feats]); axes[0].invert_yaxis()
from matplotlib.lines import Line2D
prox = [Line2D([], [], marker="o", mfc="white", color=GRAY, ls="", ms=5), Line2D([], [], marker="s", color=GRAY, ls="", ms=5)]
fig.legend(prox, ["Cross-validated cold-miRNA", "Prospective: new miRNAs in HMDD v4.0"], loc="upper center", ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("figures/fig4_prospective.png"); fig.savefig("figures/fig4_prospective.pdf")
print("fig4 ok")
