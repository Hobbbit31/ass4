# Leaderboard Accelerator Repo Skeleton

This single-file repo blueprint contains a baseline Python implementation, a C++ skeleton for optimized submission, Makefile, run scripts, submission script, auto-scorer, and a report template. Save files below into the repository structure described and run according to instructions in README.

--- FILE: README.md ---
# Accelerator Leaderboard — Repo Skeleton

This repository is a starting point for the ``Advanced Architecture Leaderboard Challenge``. It contains:

- `baseline/gemm_baseline.py` — Python baseline (multiprocessing + NumPy)
- `optimized/cpp/gemm_opt.cpp` — C++ optimized submission skeleton (multithreading + optional intrinsics)
- `Makefile` — build rules for C++ submission
- `run.sh` — script to run baseline and optimized binaries reproducibly
- `submit.sh` — packaging + metadata submission helper
- `scorer.py` — auto-scorer that runs baseline and optimized and computes speedup
- `report_template.md` — sample report template

## Quick start

1. Install dependencies: Python3, NumPy, a C++ compiler (g++), OpenMP, perf (optional).
2. Build optimized code: `make` (in repo root)
3. Run baseline: `./run.sh baseline`
4. Run optimized: `./run.sh optimized`
5. Score: `python3 scorer.py --baseline ./baseline/gemm_baseline.py --optimized ./optimized/gemm_opt`

See detailed instructions below.

--- FILE: baseline/gemm_baseline.py ---
"""
Baseline GEMM (matrix multiplication) using NumPy and multiprocessing.
This script performs a blocked multiplication and uses multiprocessing to parallelize blocks.
Usage: python3 gemm_baseline.py N num_procs
"""
import sys
import time
import numpy as np
from multiprocessing import Pool

def worker_multiply(args):
    A_block, B = args
    return A_block.dot(B)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 gemm_baseline.py N num_procs')
        sys.exit(1)
    N = int(sys.argv[1])
    P = int(sys.argv[2])

    # reproducible random
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((N, N)).astype(np.float64)
    B = rng.standard_normal((N, N)).astype(np.float64)

    # simple blocked approach: split A into P row-blocks
    blocks = [A[i::P, :] for i in range(P)]

    t0 = time.time()
    with Pool(P) as p:
        C_blocks = p.map(worker_multiply, [(blk, B) for blk in blocks])
    C = np.vstack(C_blocks)
    t1 = time.time()

    print(f'N={N} P={P} time={t1-t0:.6f} seconds')

--- FILE: optimized/cpp/gemm_opt.cpp ---
/*
C++ optimized skeleton for GEMM. Students should fill in optimizations:
- OpenMP parallelization
- Blocking / tiling
- SIMD intrinsics (AVX2 / AVX512)
- NUMA-aware allocations

Usage: ./gemm_opt N num_threads
*/
#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

static void matmul_naive(const double* A, const double* B, double* C, int N) {
    // simple triple-loop (row-major assumed)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = A[i*N + k];
            for (int j = 0; j < N; ++j) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    omp_set_num_threads(T);

    vector<double> A(N*N), B(N*N), C(N*N);
    // reproducible RNG
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i=0;i<N*N;i++) { A[i] = dist(rng); B[i] = dist(rng); C[i]=0.0; }

    double t0 = omp_get_wtime();
    matmul_naive(A.data(), B.data(), C.data(), N);
    double t1 = omp_get_wtime();
    cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

    // simple checksum to validate
    double s = 0; for (int i=0;i<N*N;i++) s += C[i];
    cout << "checksum=" << s << "\n";
    return 0;
}

--- FILE: Makefile ---
CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp -DNDEBUG
LDFLAGS = -fopenmp

all: optimized/gemm_opt

optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
	mkdir -p optimized
	$(CXX) $(CXXFLAGS) -o optimized/gemm_opt optimized/cpp/gemm_opt.cpp $(LDFLAGS)

clean:
	rm -f optimized/gemm_opt

--- FILE: run.sh ---
#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-baseline}
N=${2:-1024}
P=${3:-4}

if [ "$MODE" = "baseline" ]; then
    echo "Running baseline python implementation"
    python3 baseline/gemm_baseline.py $N $P
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

--- FILE: submit.sh ---
#!/usr/bin/env bash
# Package submission: creates a tarball with code + metadata
set -euo pipefail
TEAM=${1:-team_unknown}
OUTPUT=submit_${TEAM}_$(date +%Y%m%d_%H%M%S).tar.gz

# required files
tar -czf ${OUTPUT} README.md baseline optimized Makefile run.sh submit.sh scorer.py report_template.md

echo "Created submission package: ${OUTPUT}"

echo "Please upload ${OUTPUT} to the course server or submit link in the leaderboard sheet."

--- FILE: scorer.py ---
#!/usr/bin/env python3
"""
Auto-scorer: runs baseline and optimized (or provided commands) and computes speedup.
"""
import argparse
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True, help='Command to run baseline (e.g. "python3 baseline/gemm_baseline.py 1024 4")')
parser.add_argument('--optimized', required=True, help='Command to run optimized (e.g. "./optimized/gemm_opt 1024 4")')
parser.add_argument('--reps', type=int, default=3)
args = parser.parse_args()

def run_cmd(cmd):
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    t1 = time.time()
    if proc.returncode != 0:
        print('Command failed:', cmd)
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(1)
    return t1-t0, proc.stdout

# run baseline
print('Running baseline: ', args.baseline)
base_times = []
for i in range(args.reps):
    t, out = run_cmd(args.baseline)
    print(out)
    base_times.append(t)
base_med = sorted(base_times)[len(base_times)//2]

print('Running optimized: ', args.optimized)
opt_times = []
for i in range(args.reps):
    t, out = run_cmd(args.optimized)
    print(out)
    opt_times.append(t)
opt_med = sorted(opt_times)[len(opt_times)//2]

speedup = base_med / opt_med
print(f'Median baseline: {base_med:.6f}s, median optimized: {opt_med:.6f}s, speedup: {speedup:.3f}x')

--- FILE: report_template.md ---
# Report Template — Accelerator Leaderboard Submission

## Team: <team name>

## 1. Problem description
- Application: GEMM (dense matrix multiply)
- Input sizes tested: N = ...

## 2. Baseline
- Description of baseline implementation (Python multiprocessing + NumPy)
- Baseline timings and environment (CPU model, cores, OS)

## 3. Optimizations implemented
- Multithreading (how: OpenMP / tasks / partitioning)
- Cache blocking / tiling (explain block sizes and rationale)
- SIMD intrinsics used (AVX2/AVX-512) and code excerpts
- NUMA-aware allocation and thread pinning
- Prefetching and unrolling

## 4. Experimental methodology
- Commands used to run baseline and optimized
- Number of runs, how median chosen, perf counters collected

## 5. Results
- Tables and plots: runtime, GFLOPs, speedup vs baseline, scaling with threads
- Microarchitecture counters: L1/L2/L3 misses, bandwidth, IPC

## 6. Analysis
- Explain why the optimizations succeeded or failed.
- Identify bottlenecks and suggest further improvements.

## 7. Reproducibility
- How to reproduce runs (commands, environment, git hash)

--- END OF REPO SKELETON ---

