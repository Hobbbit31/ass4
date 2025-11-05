#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>

// Microkernel: compute 4x4 C block = A(4xK) * B(Kx4)
static inline void microkernel4x4(const double* A, const double* B, double* C, int lda, int ldb, int ldc, int K) {
    __m256d c00 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd();

    for (int k = 0; k < K; ++k) {
        __m256d b0 = _mm256_loadu_pd(B + k*ldb); // 4 doubles from row k of B
        __m256d a0 = _mm256_broadcast_sd(A + 0*lda + k);
        __m256d a1 = _mm256_broadcast_sd(A + 1*lda + k);
        __m256d a2 = _mm256_broadcast_sd(A + 2*lda + k);
        __m256d a3 = _mm256_broadcast_sd(A + 3*lda + k);

        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
    }

    _mm256_storeu_pd(C + 0*ldc, _mm256_add_pd(_mm256_loadu_pd(C + 0*ldc), c00));
    _mm256_storeu_pd(C + 1*ldc, _mm256_add_pd(_mm256_loadu_pd(C + 1*ldc), c10));
    _mm256_storeu_pd(C + 2*ldc, _mm256_add_pd(_mm256_loadu_pd(C + 2*ldc), c20));
    _mm256_storeu_pd(C + 3*ldc, _mm256_add_pd(_mm256_loadu_pd(C + 3*ldc), c30));
}

// Blocked GEMM with 4x4 microkernel
static void gemm_blocked_micro(const double* A, const double* B, double* C, int N, int BS) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += BS) {
        for (int jj = 0; jj < N; jj += BS) {
            int iMax = std::min(ii + BS, N);
            int jMax = std::min(jj + BS, N);

            for (int kk = 0; kk < N; kk += BS) {
                int kMax = std::min(kk + BS, N);

                for (int i = ii; i + 3 < iMax; i += 4) {
                    for (int j = jj; j + 3 < jMax; j += 4) {
                        microkernel4x4(A + i*N + kk, B + kk*N + j, C + i*N + j, N, N, N, kMax-kk);
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    omp_set_num_threads(T);

    std::vector<double> A((size_t)N*N), B((size_t)N*N), C((size_t)N*N, 0.0);

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < (size_t)N*N; ++i) {
        A[i] = dist(rng); B[i] = dist(rng);
    }

    double t0 = omp_get_wtime();
    gemm_blocked_micro(A.data(), B.data(), C.data(), N, 128); // BS=128 works well
    double t1 = omp_get_wtime();

    long double s=0;
    for (size_t i=0;i<(size_t)N*N;++i) s+=C[i];

    double gflops = (2.0 * (double)N * (double)N * (double)N) / (t1-t0) / 1e9;
    std::cout << "N="<<N<<" T="<<T<<" gemm="<<(t1-t0)<<" s GFLOPs="<<gflops
              <<" checksum="<<std::setprecision(17)<<(double)s<<"\n";
}
