#!/usr/bin/env bash
# Reproduce every experiment, analysis, table and figure (Linux/macOS, 2+ CPU cores, ~6-7 h).
# Usage: bash scripts/run_all.sh   (from the repository root; raw files in data/raw, see data/README.md)
set -e
mkdir -p results

# ---- part2
NTHREADS=1 bash -c "(python src/experiment.py --epochs 300 --out results/raw_random.csv --settings random --seeds 0) 2>&1 | sed -u 's/^/[1] /' & (python src/experiment.py --epochs 300 --out results/raw_cold.csv --settings cold_mirna cold_disease --seeds 0) 2>&1 | sed -u 's/^/[2] /' & wait"

# ---- part3
NTHREADS=1 bash -c "(python src/experiment.py --epochs 300 --out results/raw_b3_degree.csv --settings cold_mirna --models mlp --features rnafm_both ernierna_both rinalmo_both kmer align random ernierna_both+align rnafm_both+align sim --seeds 0 1 2 --train_neg degree --save_preds results/preds_degree) 2>&1 | sed -u 's/^/[1] /' & (python src/experiment.py --epochs 300 --out results/raw_b2_uniform.csv --settings cold_mirna --models mlp gnn --features ernierna_both rinalmo_both kmer align --seeds 0 --train_neg uniform ; python src/experiment.py --epochs 300 --out results/raw_b3_uniform.csv --settings cold_mirna --models mlp --features rnafm_both ernierna_both rinalmo_both kmer align random ernierna_both+align rnafm_both+align --seeds 0 1 2 --train_neg uniform) 2>&1 | sed -u 's/^/[2] /' & wait"

# ---- part4
NTHREADS=2 python src/experiment.py --epochs 300 --out results/raw_imc.csv --settings cold_mirna --models imc --features rnafm_both ernierna_both rinalmo_both kmer align random ernierna_both+align rnafm_both+align --seeds 0 1 2 --train_neg uniform

# ---- part5
python src/temporal.py --out results/temporal_popularity.csv --popularity --features x --seeds 0 1 2 3 4
NTHREADS=1 bash -c "(python src/temporal.py --epochs 300 --out results/temporal_degree.csv --features ernierna_both rnafm_both rinalmo_both align kmer random ernierna_both+align --train_neg degree --seeds 0 1 2 3 4) 2>&1 | sed -u 's/^/[1] /' & (python src/temporal.py --epochs 300 --out results/temporal_uniform.csv --features ernierna_both rnafm_both rinalmo_both align kmer random ernierna_both+align --train_neg uniform --seeds 0 1 2 3 4) 2>&1 | sed -u 's/^/[2] /' & wait"
NTHREADS=2 python src/temporal.py --epochs 300 --out results/temporal_uniform.csv --features ernierna_both rnafm_both rinalmo_both align kmer random ernierna_both+align --models imc --train_neg uniform --seeds 0 1 2 3 4

# ---- part6
python src/temporal.py --data data/processed_v34 --cut 29845432 --out results/prosp_popularity.csv --popularity --features x --seeds 0 1 2 3 4 5 6 7 8 9
NTHREADS=1 bash -c "(python src/temporal.py --epochs 300 --data data/processed_v34 --cut 29845432 --out results/prosp_degree.csv --features ernierna_both rnafm_both rinalmo_both align kmer random ernierna_both+align --train_neg degree --seeds 0 1 2 3 4 5 6 7 8 9) 2>&1 | sed -u 's/^/[1] /' & (python src/temporal.py --epochs 300 --data data/processed_v34 --cut 29845432 --out results/prosp_uniform.csv --features ernierna_both rnafm_both rinalmo_both align kmer random ernierna_both+align --train_neg uniform --seeds 0 1 2 3 4 5 6 7 8 9) 2>&1 | sed -u 's/^/[2] /' & wait"
NTHREADS=2 python src/temporal.py --epochs 300 --data data/processed_v34 --cut 29845432 --out results/prosp_uniform.csv --features ernierna_both rnafm_both rinalmo_both align kmer random ernierna_both+align --models imc --train_neg uniform --seeds 0 1 2 3 4 5 6 7 8 9

# ---- part7
python src/sanity_baselines.py data/processed
python src/analyze.py
python src/extra_analysis.py
python src/build_confounders.py data/raw data/processed
python src/confounder_analysis.py
python src/covariate_probe.py
python src/coreport.py data/raw
python src/analyze_temporal.py
python src/make_tables.py
python src/make_figures.py
