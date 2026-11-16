"""Combine HMDD v3 (literature to 2018) and HMDD v4.0 new records (2019-2023) into one file for a
prospective temporal validation. v4 disease names are mapped to the v3 disease vocabulary by exact
name or by an identical MeSH descriptor (via MeSH entry terms); v4 records whose disease cannot be
mapped are dropped, so every test disease has training data (pure miRNA-side evaluation).
  python src/build_v34_raw.py <raw_dir> <processed_v3_dir> <out_raw_dir>"""
import sys, os, xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
RAW, PROC, OUT = map(Path, sys.argv[1:4]); OUT.mkdir(parents=True, exist_ok=True)
term2ui = {}
for _, el in ET.iterparse(RAW / "desc2026.xml", events=("end",)):
    if el.tag != "DescriptorRecord": continue
    ui = el.findtext("DescriptorUI")
    for t in el.iter("Term"):
        s = t.findtext("String")
        if s: term2ui.setdefault(s.lower(), ui)
    el.clear()
v3 = pd.read_csv(RAW / "hmdd_associations.txt", sep="\t", dtype=str, encoding="latin1")
v4 = pd.read_csv(RAW / "alldata_v4.txt", sep="\t", dtype=str, encoding="latin1")
dis = pd.read_csv(PROC / "diseases.csv")
names = set(dis.disease); ui2name = {u: d for d, u in zip(dis.disease, dis.mesh_ui) if isinstance(u, str)}
mesh_of = dict(zip(dis.disease, dis.mesh_ui))
def map_d(x):
    x = str(x).strip()
    if x in names: return x, "exact"
    u = term2ui.get(x.lower())
    if u in ui2name: return ui2name[u], "mesh"
    return None, "none"
m = v4.disease.map(map_d)
v4["disease_v3"] = [a for a, _ in m]; how = pd.Series([b for _, b in m])
print("v4 records:", len(v4), "| disease mapping:", how.value_counts().to_dict())
v4 = v4[v4.disease_v3.notna()]
new = pd.DataFrame({"category": v4.code, "mir": v4.miRNA, "disease": v4.disease_v3, "pmid": v4.PMID,
                    "mesh": v4.disease_v3.map(mesh_of), "source": "v4"})
old = v3.assign(source="v3")[["category", "mir", "disease", "pmid", "mesh", "source"]]
comb = pd.concat([old, new], ignore_index=True)
comb.to_csv(OUT / "hmdd_associations.txt", sep="\t", index=False)
for f in ["desc2026.xml", "mature.fa", "hairpin.fa"]:
    if not (OUT / f).exists(): os.symlink((RAW / f).resolve(), OUT / f)
print("combined records:", len(comb), "(v4 kept:", len(new), ")")
