# Data

`processed/` and `processed_v34/` are included and are all that the experiments need.
To rebuild them, or to run the co-reporting and confounder analyses, put these raw files in `data/raw/`:

| File | Source |
|---|---|
| `hmdd_associations.txt` | HMDD v3.0 association file (http://www.cuilab.cn/hmdd), tab-separated with columns `category, mir, disease, pmid, ..., mesh` (MeSH descriptor of the disease) |
| `alldata_v4.txt` | HMDD v4.0 "all data" download (columns `code, PMID, miRNA, disease, description`) |
| `mature.fa`, `hairpin.fa` | miRBase release 22.1 (https://www.mirbase.org/download/) |
| `desc2026.xml` | MeSH 2026 descriptors (https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/) – only for rebuilding |

RNA language-model embeddings are produced by `notebooks/01_extract_rna_lm_embeddings.ipynb`
(output `*_mature/hairpin_embeddings.npy` and `*_names.csv`; put RNA-FM files in `data/emb/` and all models in `data/emb_lm/`).

## Contents of `processed/`
| File | Shape | Description |
|---|---|---|
| `assoc.npy` | 903 x 834 | binary miRNA–disease association matrix (HMDD v3.0) |
| `mirnas.csv`, `diseases.csv` | | row / column names; disease MeSH mapping |
| `dis_sem_sim.npy` | 834 x 834 | Wang MeSH semantic similarity |
| `emb_{rnafm,ernierna,rinalmo}_{mature,hairpin}.npy` | 903 x d | frozen LM embeddings |
| `kmer.npy` | 903 x 426 | k-mer composition |
| `seqsim_align.npy` | 903 x 1806 | Smith–Waterman similarity profile (mature + hairpin) |
| `first_pmid.npy` | 903 x 834 | earliest PMID supporting each association |
| `mirna_covariates.csv` | | miRBase discovery order, conservation, lengths |
