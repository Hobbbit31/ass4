# Usage: python3 gemm_baseline.py N num_procs
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
