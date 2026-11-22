#!/usr/bin/env bash
# Reproducible entry point for the Hybrid LinUCB aligned report.
# Requires: uv (https://docs.astral.sh/uv/) with Python 3.13.2 installed.
set -euo pipefail
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_CORETYPE=Haswell
export PYTHONHASHSEED=0

UV="${UV:-$HOME/.local/bin/uv}"

# All Python invocations use the frozen lockfile; no network at run time.
"$UV" run --frozen python reproduction/test_experiments.py -v
"$UV" run --frozen python reproduction/experiments.py --output core --sim-seeds 10 --sim-horizon 3000
"$UV" run --frozen python extend_and_audit.py --yahoo-project reproduction
"$UV" run --frozen python sparsity/run_sparsity.py
"$UV" run --frozen python make_figures.py
"$UV" run --frozen python verify_consistency.py

echo "ALL EXPERIMENTS COMPLETE"
