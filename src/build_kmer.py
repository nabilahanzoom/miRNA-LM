"""k-mer composition baseline features, aligned with data/processed/mirnas.csv.

mature: k=1..4 frequencies (340 dims), hairpin: k=1..3 (84 dims), + log lengths (2).
Uses the same name mapping rules as build_dataset.py. Also checks that the RNA-FM
name files are in the same order as the FASTA records (embedding alignment check).
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW, EMB, PROC = map(Path, sys.argv[1:4])


def read_fasta(path):
    recs, name, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                recs.append((name, "".join(buf)))
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line)
    if name:
        recs.append((name, "".join(buf)))
    return [(n, s.upper().replace("T", "U")) for n, s in recs if n.startswith("hsa-")]


def kmer_vec(seq, ks):
    out = []
    for k in ks:
        vocab = {"".join(p): i for i, p in enumerate(itertools.product("ACGU", repeat=k))}
        v = np.zeros(len(vocab), np.float32)
        for i in range(len(seq) - k + 1):
            j = vocab.get(seq[i:i + k])
            if j is not None:
                v[j] += 1
        out.append(v / max(v.sum(), 1))
    return np.concatenate(out + [np.array([np.log(len(seq))], np.float32)])


mt = read_fasta(RAW / "mature.fa")
hp = read_fasta(RAW / "hairpin.fa")
for tag, recs in [("mature", mt), ("hairpin", hp)]:
    names = pd.read_csv(EMB / f"rnafm_{tag}_names.csv")
    ok = names.name.tolist() == [n for n, _ in recs]
    ok_len = (names.length.values == np.array([min(len(s), 1022) for _, s in recs])).all()
    print(f"{tag}: {len(recs)} records; RNA-FM order matches FASTA: {ok}; lengths match: {ok_len}")
    assert ok and ok_len

K_mt = np.stack([kmer_vec(s, [1, 2, 3, 4]) for _, s in mt])
K_hp = np.stack([kmer_vec(s, [1, 2, 3]) for _, s in hp])
hp_idx = {n.lower(): i for i, (n, _) in enumerate(hp)}
mt_low = [n.lower() for n, _ in mt]

mir = pd.read_csv(PROC / "mirnas.csv")
X = np.zeros((len(mir), K_mt.shape[1] + K_hp.shape[1]), np.float32)
for r, m in enumerate(mir.mirna):
    if m in hp_idx:
        a = [hp_idx[m]]
    else:
        pat = re.compile(re.escape(m) + r"(-\d+)?$")
        a = [i for k, i in hp_idx.items() if pat.match(k)]
    pat = re.compile(re.escape(m) + r"(-\d+)?(-[35]p)?$")
    b = [i for i, k in enumerate(mt_low) if pat.match(k)]
    if b:
        X[r, :K_mt.shape[1]] = K_mt[b].mean(0)
    if a:
        X[r, K_mt.shape[1]:] = K_hp[a].mean(0)
    assert bool(a) == mir.has_hairpin[r] and bool(b) == mir.has_mature[r], m
np.save(PROC / "kmer.npy", X)
print("kmer features:", X.shape)
