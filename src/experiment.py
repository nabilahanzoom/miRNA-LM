"""RNA language-model features vs. association-derived similarity for
miRNA-disease association prediction under random and cold-start splits.

Usage:
  python experiment.py --data data/processed --out results/raw.csv \
      --settings random cold_mirna cold_disease --models mlp gnn \
      --features sim rnafm_mature rnafm_hairpin rnafm_both rnafm_both+sim random \
      --seeds 0 1 2

Leakage controls
  * GIP and miRNA functional similarity are recomputed in every fold from the
    training association matrix only (test pairs zeroed; for cold-start splits
    the whole held-out rows/columns are zeroed).
  * Feature standardisation uses training-node statistics only.
  * Test negatives are excluded from the training negative pool.
  * GNN: training positives are split each epoch into message edges (80%) and
    supervision edges (20%) so the model never predicts an edge it can see.
  * Hyper-parameters are fixed a priori (no tuning on test folds).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

import os
torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))

# ----------------------------------------------------------------- data


def load(data_dir):
    d = Path(data_dir)
    D = dict(A=np.load(d / "assoc.npy").astype(np.float32),
             S=np.load(d / "dis_sem_sim.npy").astype(np.float32),
             mature=np.load(d / "emb_mature.npy"),
             hairpin=np.load(d / "emb_hairpin.npy"),
             has_mt=np.load(d / "has_mature.npy"),
             has_hp=np.load(d / "has_hairpin.npy"))
    k = d / "kmer.npy"
    if k.exists():
        D["kmer"] = np.load(k).astype(np.float32)
    for lm in ["rnafm", "ernierna", "rinalmo"]:
        for kind in ["mature", "hairpin"]:
            f = d / f"emb_{lm}_{kind}.npy"
            if f.exists():
                D[f"{lm}_{kind}"] = np.load(f)
    if "rnafm_mature" not in D:          # backward compatibility
        D["rnafm_mature"], D["rnafm_hairpin"] = D["mature"], D["hairpin"]
    a = d / "seqsim_align.npy"
    if a.exists():
        D["align"] = np.load(a).astype(np.float32)
    return D


def gip(M):
    """Gaussian interaction profile kernel over rows of M."""
    sq = (M ** 2).sum(1)
    norm = sq.mean()
    g = 1.0 / norm if norm > 0 else 1.0
    dist = sq[:, None] + sq[None, :] - 2 * M @ M.T
    return np.exp(-g * np.maximum(dist, 0)).astype(np.float32)


def functional_sim(A, S):
    """miRNA functional similarity (Wang et al. 2010) from association matrix A."""
    n = A.shape[0]
    deg = A.sum(1)
    MaxS = np.zeros((n, S.shape[0]), np.float32)
    for i in range(n):
        idx = np.flatnonzero(A[i])
        if len(idx):
            MaxS[i] = S[idx].max(0)
    T = MaxS @ A.T                       # T[i, j] = sum_{d in Dj} max_{d' in Di} S
    den = deg[:, None] + deg[None, :]
    FS = np.where(den > 0, (T + T.T) / np.maximum(den, 1), 0).astype(np.float32)
    np.fill_diagonal(FS, 1.0)
    return FS


def _degree_matched_neg(A, tp, region_ok, axis, deg, rng):
    """For each positive (m, d) draw a negative sharing the same miRNA (axis=1 resample
    disease) or the same disease (axis=0 resample miRNA), the resampled entity drawn
    proportionally to its training degree. Controls for popularity on both sides."""
    neg = []
    p = deg / deg.sum()
    taken = set()
    for m, d in tp:
        for _ in range(1000):
            c = rng.choice(len(p), p=p)
            pair = (m, c) if axis == 1 else (c, d)
            if A[pair] == 0 and region_ok[pair] and pair not in taken:
                taken.add(pair); neg.append(pair); break
    return np.array(neg)


def make_splits(A, setting, seed, k=5):
    """Yield (train_region, test_pos, {neg_mode: test_neg}, cold) per fold.

    train_region: pairs the model may train on (test positives and both test-negative
    sets excluded). neg modes: 'uniform' (standard 1:1) and 'degree' (popularity-matched).
    """
    rng = np.random.default_rng(seed)
    nm, nd = A.shape
    kf = KFold(k, shuffle=True, random_state=seed)
    if setting == "random":
        pos = np.argwhere(A == 1)
        neg_all = np.argwhere(A == 0)
        for tr, te in kf.split(pos):
            tp = pos[te]
            tn_u = neg_all[rng.choice(len(neg_all), len(tp), replace=False)]
            A_tr = A.copy(); A_tr[tp[:, 0], tp[:, 1]] = 0
            ok = np.ones_like(A, bool); ok[tn_u[:, 0], tn_u[:, 1]] = False
            tn_d = _degree_matched_neg(A, tp, ok, 1, A_tr.sum(0) + 1, rng)
            region = np.ones_like(A, bool)
            for t in (tp, tn_u, tn_d):
                region[t[:, 0], t[:, 1]] = False
            yield region, tp, {"uniform": tn_u, "degree": tn_d}, None
    else:
        axis = 0 if setting == "cold_mirna" else 1
        n = nm if axis == 0 else nd
        for _, te in kf.split(np.arange(n)):
            held = np.zeros(n, bool); held[te] = True
            sub = np.zeros_like(A, bool)
            if axis == 0:
                sub[held] = True
            else:
                sub[:, held] = True
            tp = np.argwhere((A == 1) & sub)
            cand = np.argwhere((A == 0) & sub)
            tn_u = cand[rng.choice(len(cand), min(len(tp), len(cand)), replace=False)]
            A_tr = A * ~sub
            ok = sub.copy(); ok[tn_u[:, 0], tn_u[:, 1]] = False
            # cold miRNA: keep miRNA, resample disease by training disease degree
            # cold disease: keep disease, resample miRNA by training miRNA degree
            deg = A_tr.sum(0) + 1 if axis == 0 else A_tr.sum(1) + 1
            tn_d = _degree_matched_neg(A, tp, ok, 1 if axis == 0 else 0, deg, rng)
            yield ~sub, tp, {"uniform": tn_u, "degree": tn_d}, (axis, held)


# ----------------------------------------------------------------- features


def standardise(X, train_rows):
    mu = X[train_rows].mean(0)
    sd = X[train_rows].std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def mirna_features(D, name, A_tr, train_mirna, seed):
    nm = A_tr.shape[0]
    blocks = []
    for part in name.split("+"):
        if part == "sim":
            blocks += [functional_sim(A_tr, D["S"]), gip(A_tr)]
        elif part.endswith(("_mature", "_hairpin")):
            blocks.append(D[part])
        elif part.endswith("_both"):
            lm = part[:-5]
            blocks += [D[f"{lm}_mature"], D[f"{lm}_hairpin"]]
        elif part == "kmer":
            blocks.append(D["kmer"])
        elif part == "align":
            blocks.append(D["align"])
        elif part == "random":
            blocks.append(np.random.default_rng(1000 + seed).standard_normal((nm, 640)).astype(np.float32))
        else:
            raise ValueError(part)
    X = np.concatenate(blocks, 1)
    return standardise(X, train_mirna)


def disease_features(D, A_tr, train_dis):
    X = np.concatenate([D["S"], gip(A_tr.T)], 1)
    return standardise(X, train_dis)


def knn_adj(X, k=10):
    """Row-normalised symmetric kNN graph from cosine similarity (no self loops)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    C = Xn @ Xn.T
    np.fill_diagonal(C, -np.inf)
    idx = np.argsort(-C, 1)[:, :k]
    M = np.zeros_like(C, dtype=np.float32)
    np.put_along_axis(M, idx, 1.0, 1)
    M = np.maximum(M, M.T)
    return M / np.maximum(M.sum(1, keepdims=True), 1)


# ----------------------------------------------------------------- models


class PairDecoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1))

    def forward(self, zm, zd):
        return self.net(torch.cat([zm, zd, zm * zd], 1)).squeeze(1)


class MLPModel(nn.Module):
    def __init__(self, fm, fd, h=128):
        super().__init__()
        self.em = nn.Sequential(nn.Linear(fm, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, h))
        self.ed = nn.Sequential(nn.Linear(fd, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, h))
        self.dec = PairDecoder(h)

    def encode(self, Xm, Xd, *_):
        return self.em(Xm), self.ed(Xd)


class DotDecoder(nn.Module):
    def forward(self, zm, zd):
        return (zm * zd).sum(1)


class IMCModel(nn.Module):
    """Inductive matrix completion (bilinear) as in IMCMDA (Chen et al., Bioinformatics 2018):
    score(m, d) = x_m^T W H^T y_d with low rank r; here trained with the same loss/negatives as the other models."""
    def __init__(self, fm, fd, r=64):
        super().__init__()
        self.W, self.H = nn.Linear(fm + 1, r, bias=False), nn.Linear(fd + 1, r, bias=False)
        self.dec = DotDecoder()

    def encode(self, Xm, Xd, *_):   # constant feature appended to both sides (standard in IMC)
        one = lambda X: torch.cat([X, torch.ones(len(X), 1)], 1)
        return self.W(one(Xm)), self.H(one(Xd))


class SAGELayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.self_m, self.mm, self.dm = nn.Linear(h, h), nn.Linear(h, h, bias=False), nn.Linear(h, h, bias=False)
        self.self_d, self.dd, self.md = nn.Linear(h, h), nn.Linear(h, h, bias=False), nn.Linear(h, h, bias=False)

    def forward(self, hm, hd, Amm, Add, Amd_m, Amd_d):
        nm = self.self_m(hm) + self.mm(Amm @ hm) + self.dm(Amd_m @ hd)
        nd = self.self_d(hd) + self.dd(Add @ hd) + self.md(Amd_d @ hm)
        return F.relu(nm), F.relu(nd)


class GNNModel(nn.Module):
    def __init__(self, fm, fd, h=128, layers=2):
        super().__init__()
        self.pm, self.pd = nn.Linear(fm, h), nn.Linear(fd, h)
        self.layers = nn.ModuleList([SAGELayer(h) for _ in range(layers)])
        self.drop = nn.Dropout(0.2)
        self.dec = PairDecoder(h)

    def encode(self, Xm, Xd, Amm, Add, Amd):
        Amd_m = Amd / Amd.sum(1, keepdim=True).clamp(min=1)
        Amd_d = Amd.T / Amd.T.sum(1, keepdim=True).clamp(min=1)
        hm, hd = F.relu(self.pm(Xm)), F.relu(self.pd(Xd))
        for L in self.layers:
            hm, hd = L(self.drop(hm), self.drop(hd), Amm, Add, Amd_m, Amd_d)
        return hm, hd


# ----------------------------------------------------------------- train / eval


def sample_degree_neg(sup, allowed, p_mir, p_dis, rng, tries=20):
    """Popularity-matched training negatives: for each supervision positive (m, d), half the
    time keep m and draw d' ~ disease degree, otherwise keep d and draw m' ~ miRNA degree.
    Only pairs in `allowed` (training region, not a known association) are accepted."""
    n = len(sup)
    keep_m = rng.random(n) < 0.5
    out = sup.copy()
    todo = np.arange(n)
    for _ in range(tries):
        if not len(todo):
            break
        km = keep_m[todo]
        cand = out[todo].copy()
        cand[km, 1] = rng.choice(len(p_dis), km.sum(), p=p_dis)
        cand[~km, 0] = rng.choice(len(p_mir), (~km).sum(), p=p_mir)
        ok = allowed[cand[:, 0], cand[:, 1]]
        out[todo[ok]] = cand[ok]
        todo = todo[~ok]
    keep = np.setdiff1d(np.arange(n), todo)
    return out[keep]


def run_fold(D, A, region, tp, tn, cold, feat, model_name, seed, epochs=300, train_neg="uniform"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nm, nd = A.shape
    A_tr = (A * region).astype(np.float32)
    train_mirna = np.ones(nm, bool)
    train_dis = np.ones(nd, bool)
    if cold is not None:
        axis, held = cold
        (train_mirna if axis == 0 else train_dis)[held] = False

    Xm = torch.tensor(mirna_features(D, feat, A_tr, train_mirna, seed))
    Xd = torch.tensor(disease_features(D, A_tr, train_dis))
    Amm = torch.tensor(knn_adj(Xm.numpy()))
    Add = torch.tensor(knn_adj(D["S"] + 1e-3 * np.eye(nd, dtype=np.float32)))

    pos = np.argwhere(A_tr == 1)
    neg_pool = np.argwhere((A_tr == 0) & region)
    allowed = (A_tr == 0) & region
    p_dis = A_tr.sum(0) + 1; p_dis = p_dis / p_dis.sum()
    p_mir = A_tr.sum(1) * train_mirna + 1e-12; p_mir = p_mir / p_mir.sum()

    Model = {"mlp": MLPModel, "gnn": GNNModel, "imc": IMCModel}[model_name]
    model = Model(Xm.shape[1], Xd.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    if model_name == "imc":
        # Inductive matrix completion with squared loss on the training matrix (IMCMDA, Chen et al. 2018),
        # solved in closed form without a rank constraint (bilinear ridge / two-step RLS):
        #   Z = (Xm' Xm + lam I)^-1 Xm' A_tr Xd (Xd' Xd + lam I)^-1,  score = x_m Z y_d.
        # Only training miRNAs (cold-miRNA) / all rows otherwise enter Xm; a constant feature is appended.
        lam = 100.0
        one = lambda X: np.hstack([X, np.ones((len(X), 1), np.float32)]).astype(np.float64)
        Xm1, Xd1 = one(Xm.numpy()), one(Xd.numpy())
        rows = train_mirna
        L = np.linalg.solve(Xm1[rows].T @ Xm1[rows] + lam * np.eye(Xm1.shape[1]), Xm1[rows].T @ A_tr[rows])
        Z = np.linalg.solve(Xd1.T @ Xd1 + lam * np.eye(Xd1.shape[1]), (L @ Xd1).T).T
        P = Xm1 @ Z @ Xd1.T
        res = {}
        for mode, tn_m in tn.items():
            tp_m, tn_m = tn_m if isinstance(tn_m, tuple) else (tp, tn_m)
            te = np.concatenate([tp_m, tn_m]); s = P[te[:, 0], te[:, 1]]
            yt = np.r_[np.ones(len(tp_m)), np.zeros(len(tn_m))]
            res[mode] = (roc_auc_score(yt, s), average_precision_score(yt, s), te, s, yt)
        return res
    for ep in range(epochs):
        model.train()
        if model_name == "gnn":
            perm = rng.permutation(len(pos))
            cut = int(0.8 * len(pos))
            msg, sup = pos[perm[:cut]], pos[perm[cut:]]
            Amd = torch.zeros(nm, nd)
            Amd[msg[:, 0], msg[:, 1]] = 1
        else:
            sup, Amd = pos, None
        if train_neg == "degree":
            neg = sample_degree_neg(sup, allowed, p_mir, p_dis, rng)
        else:
            neg = neg_pool[rng.choice(len(neg_pool), len(sup), replace=False)]
        idx = np.concatenate([sup, neg])
        y = torch.cat([torch.ones(len(sup)), torch.zeros(len(neg))])
        zm, zd = model.encode(Xm, Xd, Amm, Add, Amd)
        logit = model.dec(zm[idx[:, 0]], zd[idx[:, 1]])
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        Amd = torch.tensor(A_tr) if model_name == "gnn" else None
        zm, zd = model.encode(Xm, Xd, Amm, Add, Amd)
        res = {}
        for mode, tn_m in tn.items():
            tp_m, tn_m = tn_m if isinstance(tn_m, tuple) else (tp, tn_m)   # optional (pos, neg) per mode
            te = np.concatenate([tp_m, tn_m])
            s = torch.sigmoid(model.dec(zm[te[:, 0]], zd[te[:, 1]])).numpy()
            yt = np.r_[np.ones(len(tp_m)), np.zeros(len(tn_m))]
            res[mode] = (roc_auc_score(yt, s), average_precision_score(yt, s), te, s, yt)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/raw.csv")
    ap.add_argument("--settings", nargs="+", default=["random", "cold_mirna", "cold_disease"])
    ap.add_argument("--models", nargs="+", default=["mlp", "gnn"])
    ap.add_argument("--features", nargs="+",
                    default=["sim", "rnafm_mature", "rnafm_hairpin", "rnafm_both", "rnafm_both+sim", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--save_preds", default="")
    ap.add_argument("--train_neg", default="uniform", choices=["uniform", "degree"])
    a = ap.parse_args()

    D = load(a.data)
    A = D["A"]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        if "train_neg" not in prev:
            prev["train_neg"] = "uniform"
        done = set(map(tuple, prev[["setting", "model", "feature", "seed", "fold", "train_neg"]].astype(str).values))
    for seed in a.seeds:
        for setting in a.settings:
            for fold, (region, tp, tn, cold) in enumerate(make_splits(A, setting, seed)):
                for model in a.models:
                    for feat in a.features:
                        key = (setting, model, feat, str(seed), str(fold), a.train_neg)
                        if key in done:
                            continue
                        t0 = time.time()
                        res = run_fold(D, A, region, tp, tn, cold, feat, model, seed, a.epochs, a.train_neg)
                        sec = time.time() - t0
                        rows = []
                        for mode, (auc, aupr, te, s, yt) in res.items():
                            rows.append(dict(setting=setting, model=model, feature=feat, seed=seed, fold=fold, train_neg=a.train_neg,
                                             neg=mode, auc=auc, aupr=aupr, n_test=len(yt), sec=sec))
                            if a.save_preds:
                                p = Path(a.save_preds); p.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(p / f"{setting}_{mode}_{model}_{feat}_s{seed}_f{fold}.npz",
                                                    pairs=te, score=s, y=yt)
                        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                        print(f"{setting:12s} {model:3s} {feat:16s} s{seed} f{fold}  "
                              + "  ".join(f"{r['neg']}: AUC {r['auc']:.4f} AUPR {r['aupr']:.4f}" for r in rows)
                              + f"  ({sec:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
