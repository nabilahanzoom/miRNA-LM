"""Compare your re-run (results/) with the original results (results_original/)."""
import glob, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
k = ["setting", "model", "feature", "seed", "fold", "train_neg", "neg"]
def load(d):
    x = pd.concat([pd.read_csv(f) for f in glob.glob(f"{d}/raw*.csv")])
    x["train_neg"] = x["train_neg"].fillna("uniform")
    return x.drop_duplicates(k)
new, old = load("results"), load("results_original")
m = new.merge(old, on=k, suffixes=("_new", "_old"))
print(f"=== Cross-validation: {len(m)} matched runs (yours {len(new)}, original {len(old)})")
g = m[m.setting == "cold_mirna"].groupby(["model", "train_neg", "neg", "feature"])[["auc_old", "auc_new"]].mean()
g["diff"] = g.auc_new - g.auc_old; print(g.round(4).to_string())
print(f"max |diff| per run = {(m.auc_new - m.auc_old).abs().max():.4f}, mean = {(m.auc_new - m.auc_old).abs().mean():.4f}")
for f in ["temporal_summary.csv"]:
    a = pd.read_csv(f"results/{f}"); b = pd.read_csv(f"results_original/{f}")
    kk = ["split", "model", "train_neg", "feature", "neg"]
    t = a.merge(b, on=kk, suffixes=("_new", "_old"))[kk + ["auc_mean_old", "auc_mean_new"]]
    t["diff"] = t.auc_mean_new - t.auc_mean_old
    print("\n=== Temporal / prospective (emerging miRNAs)")
    print(t[t.neg.str.startswith("tcold")].round(4).to_string(index=False))
    print(f"max |diff| = {t['diff'].abs().max():.4f}")
