#!/usr/bin/env bash
# Rebuild data/processed and data/processed_v34 from raw files (optional; processed data are included).
# Needs data/raw/{hmdd_associations.txt, alldata_v4.txt, mature.fa, hairpin.fa, desc2026.xml}
# and RNA-LM embeddings (notebooks/01_extract_rna_lm_embeddings.ipynb) in data/emb (rnafm_*) and data/emb_lm (all models).
set -e
python src/build_dataset.py data/raw data/emb data/processed
python src/build_kmer.py data/raw data/emb data/processed
python src/build_lm_features.py data/emb_lm data/processed
python src/build_align_sim.py data/raw data/processed
python src/build_temporal.py data/raw data/processed
python src/build_v34_raw.py data/raw data/processed data/raw_v34
python src/build_dataset.py data/raw_v34 data/emb data/processed_v34
python src/build_kmer.py data/raw_v34 data/emb data/processed_v34
python src/build_lm_features.py data/emb_lm data/processed_v34
python src/build_align_sim.py data/raw_v34 data/processed_v34
python src/build_temporal.py data/raw_v34 data/processed_v34
python src/build_confounders.py data/raw data/processed
