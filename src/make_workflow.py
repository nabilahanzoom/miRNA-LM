"""Study workflow schematic."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, GRAY, VIOLET, GREEN = "#2a78d6", "#eb6834", "#8a8984", "#6250d6", "#1baf7a"
tint = lambda c, a=0.10: matplotlib.colors.to_rgba(c, a)
W, H = 100, 56
fig = plt.figure(figsize=(7.4, 7.4 * H / W)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
COLS = [
 ("1  Data", 18.3, [("HMDD v3.0", ["literature to 2018", "903 miRNAs, 834 diseases", "15,659 associations"], GRAY),
                    ("HMDD v4.0", ["records 2019–2023", "443 new miRNAs"], GRAY),
                    ("miRBase", ["mature + hairpin sequences"], GRAY),
                    ("MeSH 2026", ["disease semantic similarity"], GRAY)]),
 ("2  Representations", 18.6, [("RNA language models", ["RNA-FM (640-d)", "ERNIE-RNA (768-d)", "RiNALMo-giga (1280-d)", "frozen, mean-pooled"], BLUE),
                    ("Sequence baselines", ["Smith–Waterman profile", "k-mer composition", "LM + alignment fusion"], ORANGE),
                    ("Controls", ["random features", "popularity only", "FS + GIP (classical)"], GRAY)]),
 ("3  Predictors", 16.3, [("Models", ["MLP pair model", "heterogeneous GNN", "IMC (bilinear ridge)"], VIOLET),
                    ("Training negatives", ["uniform", "popularity-debiased"], VIOLET),
                    ("Leakage control", ["train-only features,", "graphs, scaling;", "fixed hyperparameters"], VIOLET)]),
 ("4  Evaluation", 18.2, [("Cross-validation", ["random / cold-miRNA /", "cold-disease, 3×5 folds"], GREEN),
                    ("Test negatives", ["uniform (standard)", "popularity-matched"], GREEN),
                    ("Temporal validation", ["within v3.0 (cut 2014)", "prospective v3.0 → v4.0", "443 emerging miRNAs"], GREEN)]),
 ("5  Mechanisms", 16.6, [("Homology strata", ["identity to nearest", "training miRNA"], ORANGE),
                    ("Popularity probe", ["ridge → miRNA degree;", "age, conservation"], BLUE),
                    ("Co-reporting", ["homolog, same disease,", "same publication"], GRAY)]),
]
gap = (W - 2 - sum(c[1] for c in COLS)) / (len(COLS) - 1)
top, bottom = 51.5, 9.5
x = 1; spans = []
for title, w, boxes in COLS:
    ax.text(x + w / 2, 54, title, fontsize=7.8, weight="bold", color=INK, ha="center", va="center")
    hs = [4.6 + 2.45 * len(l) for _, l, _ in boxes]
    vgap = (top - bottom - sum(hs)) / (len(boxes) - 1)
    y = top
    for (bt, lines, col), h in zip(boxes, hs):
        y -= h
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=tint(col), ec=col, lw=0.9))
        ax.text(x + 1.1, y + h - 1.2, bt, fontsize=6.7, weight="bold", color=INK, va="top")
        ax.text(x + 1.1, y + h - 3.9, "\n".join(lines), fontsize=6.0, color=MUTED, va="top", linespacing=1.4)
        y -= vgap
    spans.append((x, w)); x += w + gap
for (xa, wa), (xb, _) in zip(spans[:-1], spans[1:]):
    ax.add_patch(FancyArrowPatch((xa + wa + 0.4, (top + bottom) / 2), (xb - 0.4, (top + bottom) / 2),
                                 arrowstyle="-|>", mutation_scale=10, lw=1.2, color=LINE))
ax.add_patch(FancyBboxPatch((1, 1), W - 2, 6.6, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=INK, lw=0.9))
ax.text(W / 2, 4.3, "Question: do RNA language-model embeddings carry miRNA–disease signal beyond sequence similarity and popularity\n"
        "when evaluated with popularity-matched negatives and on miRNAs reported in the future?",
        fontsize=6.6, color=INK, ha="center", va="center", linespacing=1.4)
fig.savefig("figures/fig0_workflow.pdf"); fig.savefig("figures/fig0_workflow.png")
print("ok")
