"""Alignment-based miRNA sequence similarity (Smith-Waterman, normalised).

sim(a, b) = SW(a, b) / sqrt(SW(a, a) * SW(b, b)); for miRNAs mapped to several records
the maximum over record pairs is used. Mature and hairpin matrices are concatenated
column-wise -> data/processed/seqsim_align.npy  (903 x 1806). Label-free.
Also writes seqsim_mature_max.npy (mature similarity only, used for homology strata).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align

RAW, PROC = Path(sys.argv[1]), Path(sys.argv[2])


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
    return [(n.lower(), s.upper().replace("T", "U")) for n, s in recs if n.startswith("hsa-")]


al = Align.PairwiseAligner(mode="local", match_score=2, mismatch_score=-1,
                           open_gap_score=-2, extend_gap_score=-1)
mir = pd.read_csv(PROC / "mirnas.csv")
out = []
for kind, fa in [("mature", "mature.fa"), ("hairpin", "hairpin.fa")]:
    recs = read_fasta(RAW / fa)
    names = [n for n, _ in recs]
    seqs = []
    for m in mir.mirna:
        if kind == "hairpin":
            pat = re.compile(re.escape(m) + r"(-\d+)?$")
            a = [s for n, s in recs if n == m] or [s for n, s in recs if pat.match(n)]
        else:
            pat = re.compile(re.escape(m) + r"(-\d+)?(-[35]p)?$")
            a = [s for n, s in recs if pat.match(n)]
        seqs.append(sorted(set(a)))
    uniq = sorted({s for ss in seqs for s in ss})
    ui = {s: i for i, s in enumerate(uniq)}
    self_sc = np.array([al.score(s, s) for s in uniq])
    U = np.eye(len(uniq), dtype=np.float32)
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            v = al.score(uniq[i], uniq[j]) / np.sqrt(self_sc[i] * self_sc[j])
            U[i, j] = U[j, i] = v
        if i % 200 == 0:
            print(kind, i, "/", len(uniq), flush=True)
    n = len(mir)
    M = np.zeros((n, n), np.float32)
    for i in range(n):
        if not seqs[i]:
            continue
        ii = [ui[s] for s in seqs[i]]
        for j in range(n):
            if seqs[j]:
                M[i, j] = U[np.ix_(ii, [ui[s] for s in seqs[j]])].max()
    np.save(PROC / f"seqsim_{kind}_max.npy", M)
    out.append(M)
    print(kind, "done", M.shape, "mean off-diag", M[~np.eye(n, dtype=bool)].mean().round(3))
np.save(PROC / "seqsim_align.npy", np.concatenate(out, 1))
