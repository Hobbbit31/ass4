#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-baseline}
N=${2:-1000}
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


# === BEST CONFIG ===
# KC=128 MC=256 NC=512 PREF=64
# mean GFLOPs=130.050613 stdev=0.000000 mean_time_s=0.015379
# trials (GFLOPs): 130.050613 
# checksums: 4950.575779 