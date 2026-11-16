"""Robustness experiments for cold-miRNA prediction.

* three test-negative sets per positive (m, d), all from held-out miRNA rows:
    uniform  - random unknown pair (standard)
    degree   - same miRNA, disease drawn proportionally to training degree (as in experiment.py)
    exact    - degree-preserving: disease endpoints of the test positives shuffled, so each miRNA and each
               disease appears equally often among positives and negatives (popularity-only AUROC ~0.5)
* splits:
    kfold    - 5-fold over miRNAs (identical folds/negatives to experiment.py; exact set added)
    group    - 5-fold GroupKFold over miRNA groups (e.g. miRBase families, seed families),
               so that no group has members in both training and test
* saves per-pair predictions for bootstrap CIs.

  python src/robust.py --split kfold --out results/robust_kfold_degree.csv --train_neg degree --features ...
  python src/robust.py --split group --groups data/processed/mirna_family.csv --out ...
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from experiment import load, run_fold, _degree_matched_neg


def exact_degree_neg(A, tp, region_ok, rng, tries=5000):
    """Degree-preserving negatives: the disease endpoints of the test positives are permuted, so every
    miRNA and every disease occurs exactly as often among negatives as among positives. Invalid pairs
    (known associations, outside the allowed region, duplicates) are repaired by swapping disease
    endpoints with random partners when both resulting pairs are valid. Any score that depends only on
    miRNA popularity or only on disease popularity then has AUROC 0.5."""
    m, d = tp[:, 0].copy(), tp[:, 1].copy()
    d = d[rng.permutation(len(d))]
    used = {}
    for i in range(len(m)):
        used[(m[i], d[i])] = used.get((m[i], d[i]), 0) + 1
    ok = lambda mm, dd: A[mm, dd] == 0 and region_ok[mm, dd]
    def bad(i): return not ok(m[i], d[i]) or used[(m[i], d[i])] > 1
    for i in range(len(m)):
        if not bad(i):
            continue
        for _ in range(tries):
            j = rng.integers(len(m))
            if j == i or d[j] == d[i]:
                continue
            p1, p2 = (m[i], d[j]), (m[j], d[i])
            if ok(*p1) and ok(*p2) and used.get(p1, 0) == 0 and used.get(p2, 0) == 0 and not bad(j) or \
               (ok(*p1) and ok(*p2) and used.get(p1, 0) == 0 and used.get(p2, 0) == 0):
                for q in ((m[i], d[i]), (m[j], d[j])):
                    used[q] -= 1
                d[i], d[j] = d[j], d[i]
                for q in ((m[i], d[i]), (m[j], d[j])):
                    used[q] = used.get(q, 0) + 1
                break
    keep, seen = [], set()
    for i in range(len(m)):
        q = (m[i], d[i])
        if ok(*q) and q not in seen:
            seen.add(q); keep.append(i)
    return np.c_[m[keep], d[keep]], len(tp) - len(keep)


def splits(A, how, seed, groups=None, k=5):
    rng = np.random.default_rng(seed)
    rng_x = np.random.default_rng(10_000 + seed)            # separate stream: old negatives unchanged
    nm = A.shape[0]
    if how == "kfold":
        it = KFold(k, shuffle=True, random_state=seed).split(np.arange(nm))
    else:
        g = groups.copy()
        perm = np.random.default_rng(seed).permutation(np.unique(g))   # shuffle group->fold assignment per seed
        remap = {v: i for i, v in enumerate(perm)}
        it = GroupKFold(k).split(np.arange(nm), groups=np.array([remap[v] for v in g]))
    for _, te in it:
        held = np.zeros(nm, bool); held[te] = True
        sub = np.zeros_like(A, bool); sub[held] = True
        tp = np.argwhere((A == 1) & sub)
        cand = np.argwhere((A == 0) & sub)
        tn_u = cand[rng.choice(len(cand), min(len(tp), len(cand)), replace=False)]
        A_tr = A * ~sub
        ok = sub.copy(); ok[tn_u[:, 0], tn_u[:, 1]] = False
        deg = A_tr.sum(0) + 1
        tn_d = _degree_matched_neg(A, tp, ok, 1, deg, rng)
        ok2 = ok.copy(); ok2[tn_d[:, 0], tn_d[:, 1]] = False
        tn_x, dropped = exact_degree_neg(A, tp, ok2, rng_x)
        yield ~sub, tp, {"uniform": tn_u, "degree": tn_d, "exact": tn_x}, (0, held), dropped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed"); ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["kfold", "group"], default="kfold")
    ap.add_argument("--groups", default=""); ap.add_argument("--group_col", default="group")
    ap.add_argument("--features", nargs="+", required=True); ap.add_argument("--models", nargs="+", default=["mlp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train_neg", default="degree"); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--save_preds", default="")
    a = ap.parse_args()
    D = load(a.data); A = D["A"]
    groups = None
    if a.split == "group":
        gdf = pd.read_csv(a.groups); groups = np.array([str(v) if isinstance(v, str) else 'single_' + m for v, m in zip(gdf[a.group_col], gdf['mirna'])])   # missing seed -> own group
        assert len(groups) == A.shape[0]
    out = Path(a.out); done = set()
    if out.exists():
        done = set(map(tuple, pd.read_csv(out)[["model", "feature", "seed", "fold"]].astype(str).values))
    for seed in a.seeds:
        for fold, (region, tp, tn, cold, gap) in enumerate(splits(A, a.split, seed, groups)):
            if seed == a.seeds[0]:
                print(f"fold {fold}: test miRNAs {cold[1].sum()}, positives {len(tp)}, degree-preserving negatives {len(tn['exact'])} (dropped {gap})", flush=True)
            for model in a.models:
                for feat in a.features:
                    if (model, feat, str(seed), str(fold)) in done:
                        continue
                    t0 = time.time()
                    res = run_fold(D, A, region, tp, tn, cold, feat, model, seed, a.epochs, a.train_neg)
                    rows = []
                    for mode, (auc, aupr, te, s, yt) in res.items():
                        rows.append(dict(split=a.split, model=model, feature=feat, seed=seed, fold=fold, train_neg=a.train_neg,
                                         neg=mode, auc=auc, aupr=aupr, n_test=len(yt), exact_dropped=int(gap)))
                        if a.save_preds:
                            p = Path(a.save_preds); p.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(p / f"{a.split}_{mode}_{model}_{feat}_{a.train_neg}_s{seed}_f{fold}.npz",
                                                pairs=te, score=s, y=yt)
                    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                    print(model, feat, seed, fold, " ".join(f"{r['neg']}={r['auc']:.3f}" for r in rows), f"{time.time()-t0:.0f}s", flush=True)
