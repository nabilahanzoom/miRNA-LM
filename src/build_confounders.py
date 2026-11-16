"""miRNA-level covariates that could explain why sequence representations predict research popularity.
discovery order : smallest miRBase accession (MI/MIMAT number) among mapped records (earlier = older)
conservation    : number of species (other than human) in miRBase with an identical mature sequence,
                  and with an identical seed (nt 2-8)
length          : mean mature / hairpin length;  availability flags
Writes data/processed/mirna_covariates.csv"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
RAW, PROC = Path(sys.argv[1]), Path(sys.argv[2])

def fasta(p):
    out, h, buf = [], None, []
    for l in open(p):
        l = l.strip()
        if l.startswith(">"):
            if h: out.append((h, "".join(buf)))
            h, buf = l[1:].split(), []
        else: buf.append(l)
    out.append((h, "".join(buf)))
    return [(h[0].lower(), h[1], s.upper().replace("T", "U")) for h, s in out]

mt, hp = fasta(RAW / "mature.fa"), fasta(RAW / "hairpin.fa")
seq_species, seed_species = {}, {}
for n, _, s in mt:
    sp = n.split("-")[0]
    if sp == "hsa": continue
    seq_species.setdefault(s, set()).add(sp); seed_species.setdefault(s[1:8], set()).add(sp)
hmt = [(n, a, s) for n, a, s in mt if n.startswith("hsa-")]
hhp = [(n, a, s) for n, a, s in hp if n.startswith("hsa-")]
num = lambda a: int(re.sub(r"\D", "", a))
rows = []
for m in pd.read_csv(PROC / "mirnas.csv").mirna:
    p1 = re.compile(re.escape(m) + r"(-\d+)?(-[35]p)?$"); p2 = re.compile(re.escape(m) + r"(-\d+)?$")
    a = [r for r in hmt if p1.match(r[0])]
    b = [r for r in hhp if r[0] == m] or [r for r in hhp if p2.match(r[0])]
    rows.append(dict(mirna=m,
        acc_hairpin=min([num(x[1]) for x in b], default=np.nan),
        acc_mature=min([num(x[1]) for x in a], default=np.nan),
        cons_seq=max([len(seq_species.get(x[2], ())) for x in a], default=np.nan),
        cons_seed=max([len(seed_species.get(x[2][1:8], ())) for x in a], default=np.nan),
        len_mature=np.mean([len(x[2]) for x in a]) if a else np.nan,
        len_hairpin=np.mean([len(x[2]) for x in b]) if b else np.nan,
        n_records=len(a) + len(b)))
c = pd.DataFrame(rows); c.to_csv(PROC / "mirna_covariates.csv", index=False)
print(c.describe().round(1).to_string())
