#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-baseline}
N=${2:-8000}
P=${3:-8}

if [ "$MODE" = "baseline" ]; then
    echo "Running baseline python implementation"
    python3 baseline/gemm_baseline.py $N $P
    # python3 baseline/run.py $N $P

elif [ "$MODE" = "optimized" ]; then
    echo "Running optimized binary"
    if [ ! -x optimized/gemm_opt ]; then
        echo "Optimized binary not found. Run make first."
        exit 1
    fi
    optimized/gemm_opt $N $P
else
    echo "Unknown mode: $MODE"
    exit 1
fi
