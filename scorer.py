#!/usr/bin/env python3
"""
Auto-scorer: runs baseline and optimized (or provided commands) and computes speedup.
"""


# 5. Score: python3 scorer.py --baseline ./baseline/gemm_baseline.py --optimized ./optimized/cpp/gemm_opt

import argparse
import subprocess
import time

parser = argparse.ArgumentParser() 
parser.add_argument('--baseline', required=True, help='Command to run baseline (e.g. "python3 baseline/gemm_baseline.py 1024 4")')
parser.add_argument('--optimized', required=True, help='Command to run optimized (e.g. "./optimized/gemm_opt.exe 1024 4")') 
parser.add_argument('--reps', type=int, default=10) # number of repetitions for median
args = parser.parse_args()

def run_cmd(cmd):
    t0 = time.time()
    cmd = cmd + " 4000 8" # Example arguments for matrix size and block size
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
    t, out = run_cmd("python " + args.baseline)
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