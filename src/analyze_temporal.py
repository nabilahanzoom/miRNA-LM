"""Summaries and paired tests for the within-v3 temporal split and the prospective v3->v4 validation."""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
out = []
for tag, files in {"within_v3": ["temporal_degree", "temporal_uniform", "temporal_popularity"],
                   "prospective": ["prosp_degree", "prosp_uniform", "prosp_popularity"]}.items():
    d = pd.concat([pd.read_csv(f"results/{f}.csv") for f in files]).assign(split=tag)
    d = d.drop_duplicates(["model", "feature", "seed", "train_neg", "neg"])   # re-runs may append identical rows
    out.append(d)
d = pd.concat(out)
s = d.groupby(["split", "model", "train_neg", "feature", "neg"])[["auc", "aupr"]].agg(["mean", "std", "count"]).round(6)
s.columns = ["_".join(c) for c in s.columns]; s.reset_index().to_csv("results/temporal_summary.csv", index=False)
rows = []
for (sp, mo, tn), g in d[d.model != "none"].groupby(["split", "model", "train_neg"]):
    pop = d[(d.split == sp) & (d.feature == "popularity")]
    for neg in g.neg.unique():
        x = g[g.neg == neg].pivot_table(index="seed", columns="feature", values="auc")
        p = pop[pop.neg == neg].set_index("seed").auc
        comps = [(a, "random") for a in x.columns if a != "random"] + [(a, "align") for a in x.columns if a not in ("align", "random")]
        comps += [(a, "popularity") for a in x.columns]
        for a, b in comps:
            xb = p if b == "popularity" else x[b]
            j = x[a].dropna().index.intersection(xb.dropna().index)
            if len(j) < 5: continue
            diff = x[a][j] - xb[j]
            rows.append(dict(split=sp, model=mo, train_neg=tn, neg=neg, a=a, b=b, n=len(j), mean_a=x[a][j].mean().round(4),
                             mean_b=xb[j].mean().round(4), diff=diff.mean().round(4), wins=int((diff > 0).sum()),
                             p=wilcoxon(x[a][j], xb[j]).pvalue if (diff != 0).any() else 1.0))
pd.DataFrame(rows).to_csv("results/temporal_paired.csv", index=False)
print("ok", len(rows))
