"""Illustrated study workflow (graphical overview) built partly from the real data."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse, Polygon, Arc
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 400})
INK, MUTED, LINE, BG = "#1d1d1b", "#55544f", "#9a998f", "#f7f6f2"
BLUE, ORANGE, AQUA, YEL, MAG, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
P = "data/processed/"
A = np.load(P + "assoc.npy"); S = np.load(P + "seqsim_mature_max.npy"); E = np.load(P + "emb_ernierna_mature.npy")
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(7.4, 4.4))
fig.patch.set_facecolor("white")
W = [0.012, 0.212, 0.408, 0.604, 0.800]; PW = 0.188
STAGES = [("a", "Data", BLUE), ("b", "Representations", VIOLET), ("c", "Predictors", AQUA),
          ("d", "Fair evaluation", ORANGE), ("e", "Mechanisms", RED)]
axes = []
for x, (lab, title, col) in zip(W, STAGES):
    bg = fig.add_axes([x, 0.17, PW, 0.80]); bg.set_xlim(0, 10); bg.set_ylim(0, 20); bg.axis("off")
    bg.add_patch(FancyBboxPatch((0.05, 0.05), 9.9, 19.9, boxstyle="round,pad=0,rounding_size=0.6", fc=BG, ec="none"))
    bg.add_patch(FancyBboxPatch((0.05, 18.0), 9.9, 1.95, boxstyle="round,pad=0,rounding_size=0.6", fc=col, ec="none"))
    bg.add_patch(Rectangle((0.05, 18.0), 9.9, 0.7, fc=col, ec="none"))
    bg.text(0.6, 18.95, lab, fontsize=9, weight="bold", color="white", va="center")
    bg.text(1.5, 18.95, title, fontsize=7.2, weight="bold", color="white", va="center")
    axes.append(bg)

def sub(ax_bg, x0, y0, w, h):
    """inset axes in background-panel data coordinates"""
    pos = ax_bg.get_position()
    return fig.add_axes([pos.x0 + pos.width * x0 / 10, pos.y0 + pos.height * y0 / 20, pos.width * w / 10, pos.height * h / 20])
def cap(ax, x, y, t, fs=5.6, c=MUTED, **k): ax.text(x, y, t, fontsize=fs, color=c, ha="center", va="top", linespacing=1.3, **k)
def head(ax, x, y, t, fs=6.3): ax.text(x, y, t, fontsize=fs, color=INK, ha="center", va="center", weight="bold")

# ---------------- a: data
a = axes[0]
def cylinder(ax, cx, cy, w, h, col, label, sub_):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=col, ec="none", alpha=0.9))
    ax.add_patch(Ellipse((cx, cy - h / 2), w, 0.9, fc=col, ec="none", alpha=0.9))
    ax.add_patch(Ellipse((cx, cy + h / 2), w, 0.9, fc=matplotlib.colors.to_rgba(col, 0.55), ec="white", lw=0.8))
    for k in (1, 2): ax.add_patch(Arc((cx, cy + h / 2 - k * h / 3), w, 0.9, theta1=180, theta2=360, ec="white", lw=0.5, alpha=0.7))
    ax.text(cx, cy - 0.15, label, fontsize=5.8, color="white", weight="bold", ha="center", va="center")
    cap(ax, cx, cy - h / 2 - 0.65, sub_, fs=5.2)
cylinder(a, 2.7, 15.0, 3.6, 2.6, BLUE, "HMDD\nv3.0", "≤ 2018\n15,659 pairs")
cylinder(a, 7.3, 15.0, 3.6, 2.6, "#174f91", "HMDD\nv4.0", "2019–2023\n443 new miRNAs")
# association matrix (real data, top miRNAs x diseases)
ia = sub(a, 0.8, 6.4, 4.2, 4.0)
o_m = np.argsort(-A.sum(1))[:40]; o_d = np.argsort(-A.sum(0))[:40]
ia.imshow(A[np.ix_(o_m, o_d)], cmap=matplotlib.colors.ListedColormap(["white", BLUE]), aspect="auto", interpolation="nearest")
ia.set_xticks([]); ia.set_yticks([]); [s.set_color(LINE) for s in ia.spines.values()]
cap(a, 2.9, 6.1, "associations\n(top 40 × 40)", fs=4.9)
# hairpin drawing
def hairpin(ax, x0, y0, s=1.0, col=VIOLET):
    t = np.linspace(0, 1, 9)
    for i, tt in enumerate(t):
        ax.add_patch(Circle((x0, y0 + tt * 3.6 * s), 0.17 * s, fc=col, ec="none"))
        ax.add_patch(Circle((x0 + 0.9 * s, y0 + tt * 3.6 * s), 0.17 * s, fc=matplotlib.colors.to_rgba(col, .55), ec="none"))
        ax.plot([x0 + .17 * s, x0 + .73 * s], [y0 + tt * 3.6 * s] * 2, color=LINE, lw=0.5)
    th = np.linspace(np.pi, 0, 7)
    for tt in th: ax.add_patch(Circle((x0 + .45 * s + .62 * s * np.cos(tt), y0 + 3.6 * s + .55 * s + .5 * s * np.sin(tt)), .17 * s, fc=YEL, ec="none"))
hairpin(a, 6.9, 7.0, 0.85)
cap(a, 7.3, 6.1, "miRBase\nsequences", fs=4.9)
a.add_patch(FancyBboxPatch((0.8, 0.9), 8.4, 3.0, boxstyle="round,pad=0,rounding_size=0.4", fc="white", ec=LINE, lw=0.6))
a.text(5, 2.95, "MeSH 2026", fontsize=5.8, weight="bold", color=INK, ha="center", va="center")
for xx, yy, xx2, yy2 in [(5, 2.3, 3.3, 1.45), (5, 2.3, 5, 1.45), (5, 2.3, 6.7, 1.45)]:
    a.plot([xx, xx2], [yy, yy2], color=LINE, lw=0.6)
for xx in (3.3, 5, 6.7): a.add_patch(Circle((xx, 1.35), 0.22, fc=GREEN, ec="none"))
a.add_patch(Circle((5, 2.35), 0.22, fc=GREEN, ec="none"))

# ---------------- b: representations
b = axes[1]
b.text(5, 17.35, "hsa-miR-21-5p", fontsize=4.8, color=MUTED, ha="center", va="center")
b.text(5, 16.8, "5′-UAGCUUAUCAGACUGAUGUUGA-3′", fontsize=4.6, family="DejaVu Sans Mono", color=INK, ha="center", va="center")
for i, (nm, col) in enumerate([("RNA-FM", BLUE), ("ERNIE-RNA", VIOLET), ("RiNALMo", MAG)]):
    x = 1.0 + i * 2.85
    for k in range(4):
        b.add_patch(FancyBboxPatch((x, 13.0 + k * 0.72), 2.4, 0.52, boxstyle="round,pad=0,rounding_size=0.18", fc=matplotlib.colors.to_rgba(col, 0.35 + 0.15 * k), ec="none"))
    b.text(x + 1.2, 12.45, nm, fontsize=4.5, color=INK, ha="center", va="center", weight="bold")
    b.add_patch(FancyArrowPatch((5, 16.4), (x + 1.2, 16.0), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=LINE))
cap(b, 5, 12.0, "frozen transformers, mean-pooled", fs=5.0)
ie = sub(b, 0.8, 9.4, 8.4, 1.3)
v = E[np.argsort(-A.sum(1))[:1]][:, :96]
ie.imshow(v, cmap="RdBu_r", aspect="auto", vmin=-np.abs(v).max(), vmax=np.abs(v).max()); ie.axis("off")
cap(b, 5, 9.2, "ERNIE-RNA embedding (96 of 768 dims)", fs=4.8)
isim = sub(b, 0.8, 1.6, 3.9, 5.2)
o = np.argsort(-A.sum(1))[:45]
isim.imshow(S[np.ix_(o, o)] ** 2, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("o", ["white", ORANGE, "#8a2f0c"]), interpolation="nearest", vmin=0.1, vmax=0.9)
isim.set_xticks([]); isim.set_yticks([]); [s.set_color(LINE) for s in isim.spines.values()]
cap(b, 2.75, 1.4, "Smith–Waterman\nsimilarity", fs=5.0)
ik = sub(b, 5.6, 1.6, 3.6, 5.2)
K = np.load(P + "kmer.npy")[o[0], :16]
ik.bar(range(16), K, color=YEL, width=0.8); ik.set_xticks([]); ik.set_yticks([])
for s in ["top", "right", "left"]: ik.spines[s].set_visible(False)
ik.spines["bottom"].set_color(LINE)
cap(b, 7.4, 1.4, "k-mer composition\n+ random control", fs=5.0)

# ---------------- c: predictors
c = axes[2]
head(c, 5, 16.9, "MLP pair model")
layers = [3, 4, 4, 1]
for li, n in enumerate(layers):
    for j in range(n):
        x, y = 2.0 + li * 2.0, 14.4 + (j - (n - 1) / 2) * 0.85
        if li < len(layers) - 1:
            for j2 in range(layers[li + 1]):
                c.plot([x, x + 2.0], [y, 14.4 + (j2 - (layers[li + 1] - 1) / 2) * 0.85], color=LINE, lw=0.3, alpha=0.7)
        c.add_patch(Circle((x, y), 0.27, fc=AQUA if li < 3 else INK, ec="white", lw=0.4, zorder=3))
head(c, 5, 11.1, "Heterogeneous GNN")
mx = [(1.8, 9.6), (1.8, 8.2), (1.8, 6.8)]; dx = [(8.2, 9.9), (8.2, 8.6), (8.2, 7.3), (8.2, 6.0)]
for (x1, y1) in mx:
    for (x2, y2) in dx:
        if rng.random() < 0.45: c.plot([x1, x2], [y1, y2], color=LINE, lw=0.6)
for (x, y) in mx: c.add_patch(Circle((x, y), 0.42, fc=VIOLET, ec="white", lw=0.6, zorder=3))
for (x, y) in dx: c.add_patch(Rectangle((x - 0.38, y - 0.38), 0.76, 0.76, fc=GREEN, ec="white", lw=0.6, zorder=3))
c.plot([1.8, 1.8], [9.6, 6.8], color=VIOLET, lw=0.8, ls=":"); c.plot([8.2, 8.2], [9.9, 6.0], color=GREEN, lw=0.8, ls=":")
c.text(1.8, 5.95, "miRNA", fontsize=4.8, color=MUTED, ha="center"); c.text(8.2, 5.15, "disease", fontsize=4.8, color=MUTED, ha="center")
head(c, 5, 4.2, "IMC (bilinear)")
for i, (w_, h_, col, lab) in enumerate([(0.9, 2.1, VIOLET, "X"), (0.9, 0.9, AQUA, "W"), (0.9, 0.9, AQUA, "Hᵀ"), (2.1, 0.9, GREEN, "Yᵀ")]):
    x = 1.0 + [0, 1.5, 3.0, 4.5][i]
    c.add_patch(Rectangle((x, 1.2 + (2.1 - h_) / 2), w_, h_, fc=matplotlib.colors.to_rgba(col, 0.7), ec="none"))
    c.text(x + w_ / 2, 2.25, lab, fontsize=5.6, color="white", ha="center", va="center", weight="bold")
c.text(7.9, 2.25, "→ score", fontsize=5.4, color=INK, va="center")

# ---------------- d: evaluation
d = axes[3]
head(d, 5, 16.9, "Test negatives")
deg = np.sort(A.sum(0))[::-1][:30]
ih = sub(d, 0.8, 11.8, 8.4, 4.0)
ih.bar(range(30), deg, color=matplotlib.colors.to_rgba(ORANGE, .35), width=0.85)
idx_u = rng.choice(30, 5, replace=False); idx_d = rng.choice(30, 5, replace=False, p=deg / deg.sum())
ih.scatter(idx_u, deg[idx_u] + 25, marker="v", s=9, color=LINE, zorder=3, label="uniform")
ih.scatter(idx_d, deg[idx_d] + 25, marker="v", s=9, color=ORANGE, zorder=3, label="popularity-matched")
ih.set_xticks([]); ih.set_yticks([]); [ih.spines[s].set_visible(False) for s in ["top", "right", "left"]]
ih.spines["bottom"].set_color(LINE)
ih.legend(fontsize=4.3, frameon=False, loc="upper right", handletextpad=0.1, bbox_to_anchor=(1.02, 1.0), borderaxespad=0.1)
cap(d, 5, 11.4, "disease popularity (training degree)", fs=5.0)
head(d, 5, 9.4, "Splits")
for k in range(5):
    for j in range(5):
        d.add_patch(Rectangle((1.0 + j * 1.0, 7.6 - k * 0.0), 0.9, 0.7, fc=ORANGE if j == 2 else matplotlib.colors.to_rgba(ORANGE, .25), ec="white", lw=0.5))
d.text(6.4, 7.95, "5-fold\ncold-miRNA", fontsize=4.8, color=MUTED, va="center")
d.add_patch(FancyArrowPatch((0.9, 4.4), (9.3, 4.4), arrowstyle="-|>", mutation_scale=7, lw=1.0, color=INK))
for x, t in [(1.6, "2003"), (4.0, "2014"), (6.2, "2018"), (8.5, "2023")]:
    d.plot([x, x], [4.15, 4.65], color=INK, lw=0.8); d.text(x, 3.75, t, fontsize=4.8, color=MUTED, ha="center", va="top")
d.add_patch(Rectangle((1.6, 5.0), 4.6, 0.65, fc=BLUE, ec="none")); d.text(3.9, 5.32, "train: HMDD v3.0", fontsize=4.7, color="white", ha="center", va="center")
d.add_patch(Rectangle((6.2, 5.0), 3.1, 0.65, fc=RED, ec="none")); d.text(7.75, 5.32, "test: v4.0", fontsize=4.7, color="white", ha="center", va="center")
cap(d, 5, 2.6, "prospective validation on\n443 miRNAs first reported 2019–2023", fs=5.0)

# ---------------- e: mechanisms (real numbers)
e = axes[4]
head(e, 5, 16.9, "Gain over random features")
ig = sub(e, 3.0, 11.4, 6.4, 4.8)
_s = pd.read_csv("results/summary_all.csv"); _s = _s[(_s.setting == "cold_mirna") & (_s.model == "mlp") & (_s.train_neg == "degree") & (_s.neg == "degree")].set_index("feature").auc_mean
_t = pd.read_csv("results/temporal_summary.csv"); _t = _t[(_t.split == "prospective") & (_t.model == "mlp") & (_t.train_neg == "degree") & (_t.neg == "tcold_degree")].set_index("feature").auc_mean
vals_cv = [_s["ernierna_both"] - _s["random"], _s["align"] - _s["random"]]
vals_pr = [_t["ernierna_both"] - _t["random"], _t["align"] - _t["random"]]
print("panel e:", [round(v, 3) for v in vals_cv + vals_pr])
ig.barh([1, 0], vals_cv, color=matplotlib.colors.to_rgba(VIOLET, .35), height=0.38, label="cross-validation")
ig.barh([0.6, -0.4], vals_pr, color=RED, height=0.38, label="prospective")
ig.axvline(0, color=INK, lw=0.6); ig.set_yticks([0.8, -0.2], ["ERNIE-RNA", "alignment"], fontsize=4.6)
ig.set_xticks([0, 0.05, 0.1]); ig.tick_params(labelsize=4.5, length=2, colors=MUTED)
for s in ["top", "right", "left"]: ig.spines[s].set_visible(False)
ig.legend(fontsize=4.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, handlelength=1)
ig.set_xlabel("ΔAUROC (popularity-matched)", fontsize=4.4, color=MUTED, labelpad=1.5)
head(e, 5, 8.2, "Popularity encoded")
ip = sub(e, 1.6, 4.3, 7.6, 3.1)
names = ["ERNIE-RNA", "RNA-FM", "RiNALMo", "align", "k-mer"]
_e = pd.read_csv("results/popularity_encoding.csv").set_index("feature").spearman
rho = [_e[k] for k in ["ernierna_both", "rnafm_both", "rinalmo_both", "align", "kmer"]]
ip.bar(range(5), rho, color=[BLUE, BLUE, BLUE, ORANGE, YEL], width=0.7)
ip.set_xticks(range(5), names, rotation=35, fontsize=4.3, ha="right"); ip.set_yticks([0, .4, .8]); ip.set_ylim(0, .85)
ip.tick_params(labelsize=4.3, length=2, colors=MUTED)
for s in ["top", "right"]: ip.spines[s].set_visible(False)
ip.set_ylabel("ρ with degree", fontsize=4.4, color=MUTED, labelpad=1)
e.add_patch(FancyBboxPatch((0.8, 0.6), 8.4, 1.65, boxstyle="round,pad=0,rounding_size=0.3", fc="white", ec=LINE, lw=0.6))
e.text(5, 1.42, f"{100*pd.read_csv('results/coreporting.csv').homolog_same_disease_same_paper.iloc[0]:.0f}% of associations co-reported\nwith a homolog in the same paper", fontsize=4.7, color=INK, ha="center", va="center", linespacing=1.3)

# arrows between panels
ov = fig.add_axes([0, 0, 1, 1], zorder=10); ov.axis("off"); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
for x in W[1:]:
    ov.add_patch(FancyArrowPatch((x - 0.012, 0.57), (x + 0.001, 0.57), arrowstyle="simple,head_width=4,head_length=4", mutation_scale=1.6, color=INK))
# bottom banner
ov.add_patch(FancyBboxPatch((0.012, 0.035), 0.976, 0.10, boxstyle="round,pad=0,rounding_size=0.012", fc="white", ec=INK, lw=0.8, transform=ov.transAxes))
ov.text(0.5, 0.0825, "Under popularity-matched and prospective evaluation, RNA language models add at most ~0.01 AUROC over random\n"
        "miRNA features; their apparent advantage in standard benchmarks reflects popularity (age, conservation) and homolog co-reporting.",
        fontsize=5.9, color=INK, ha="center", va="center", linespacing=1.4)
fig.savefig("figures/fig0_workflow_v2.png"); fig.savefig("figures/fig0_workflow_v2.pdf"); fig.savefig("figures/fig0_workflow_v2.tiff", dpi=600)
print("ok")
