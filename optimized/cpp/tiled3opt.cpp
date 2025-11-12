// // #include <bits/stdc++.h>
// // #include <omp.h>
// // #include <immintrin.h>
// // using namespace std;

// // // Parameters tuned for L1/L2 cache sizes
// // constexpr int MC = 256;   // Rows of A block (L2)
// // constexpr int NC = 256;   // Cols of B block (L3)
// // constexpr int KC = 256;   // Depth block (L1)
// // constexpr int MR = 4;     // Microkernel rows
// // constexpr int NR = 4;     // Microkernel cols

// // // AVX2 4x4 microkernel
// // static inline void microkernel4x4(const double* A, const double* B, double* C,
// //                                   int lda, int ldb, int ldc) {
// //     __m256d c0 = _mm256_loadu_pd(&C[0*ldc]); // C[0][0..3]
// //     __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
// //     __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
// //     __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);

// //     for (int k = 0; k < KC; ++k) {
// //         __m256d b = _mm256_loadu_pd(&B[k*ldb]); // 4 doubles
// //         __m256d a0 = _mm256_broadcast_sd(&A[0*lda + k]);
// //         __m256d a1 = _mm256_broadcast_sd(&A[1*lda + k]);
// //         __m256d a2 = _mm256_broadcast_sd(&A[2*lda + k]);
// //         __m256d a3 = _mm256_broadcast_sd(&A[3*lda + k]);

// //         c0 = _mm256_fmadd_pd(a0, b, c0);
// //         c1 = _mm256_fmadd_pd(a1, b, c1);
// //         c2 = _mm256_fmadd_pd(a2, b, c2);
// //         c3 = _mm256_fmadd_pd(a3, b, c3);
// //     }

// //     _mm256_storeu_pd(&C[0*ldc], c0);
// //     _mm256_storeu_pd(&C[1*ldc], c1);
// //     _mm256_storeu_pd(&C[2*ldc], c2);
// //     _mm256_storeu_pd(&C[3*ldc], c3);
// // }

// // // Blocked GEMM driver
// // static void gemm_blocked(const double* A, const double* B, double* C, int N) {
// //     #pragma omp parallel for collapse(2) schedule(static)
// //     for (int jc = 0; jc < N; jc += NC) {
// //         for (int pc = 0; pc < N; pc += KC) {
// //             for (int ic = 0; ic < N; ic += MC) {
// //                 int jMax = min(jc + NC, N);
// //                 int pMax = min(pc + KC, N);
// //                 int iMax = min(ic + MC, N);

// //                 for (int i = ic; i < iMax; i += MR) {
// //                     for (int j = jc; j < jMax; j += NR) {
// //                         double* Cblock = &C[i*N + j];
// //                         const double* Ablock = &A[i*N + pc];
// //                         const double* Bblock = &B[pc*N + j];
// //                         microkernel4x4(Ablock, Bblock, Cblock, N, N, N);
// //                     }
// //                 }
// //             }
// //         }
// //     }
// // }

// // int main(int argc, char** argv) {
// //     if (argc < 3) {
// //         cerr << "Usage: " << argv[0] << " N num_threads\n";
// //         return 1;
// //     }
// //     int N = atoi(argv[1]);
// //     int T = atoi(argv[2]);
// //     if (N <= 0 || T <= 0) {
// //         cerr << "N and num_threads must be > 0\n";
// //         return 1;
// //     }

// //     omp_set_num_threads(T);

// //     vector<double> A((size_t)N*N), B((size_t)N*N), C((size_t)N*N, 0.0);
// //     std::mt19937_64 rng(12345);
// //     std::normal_distribution<double> dist(0.0, 1.0);
// //     for (size_t i = 0; i < (size_t)N*N; ++i) {
// //         A[i] = dist(rng);
// //         B[i] = dist(rng);
// //     }

// //     double t0 = omp_get_wtime();
// //     gemm_blocked(A.data(), B.data(), C.data(), N);
// //     double t1 = omp_get_wtime();
// //     double elapsed = t1 - t0;

// //     // checksum
// //     long double s = 0.0L;
// //     for (size_t i = 0; i < (size_t)N*N; ++i) s += C[i];

// //     double gflops = (2.0 * N * N * N) / (elapsed * 1e9);
// //     cout << "N=" << N << " T=" << T
// //          << " gemm=" << elapsed << " s GFLOPs=" << gflops
// //          << " checksum=" << setprecision(17) << (double)s << "\n";

// //     return 0;
// // }



// #include <bits/stdc++.h>
// #include <omp.h>
// #include <immintrin.h>
// using namespace std;

// // Blocking params (good defaults; tune for your CPU)
// constexpr int MC = 128;   // L2-friendly rows block
// constexpr int NC = 256;   // L3-friendly cols block
// constexpr int KC = 128;   // L1-friendly depth block
// constexpr int MR = 4;     // micro-kernel rows (AVX2)
// constexpr int NR = 4;     // micro-kernel cols (AVX2)

// // Prefetch depth (tuneable)
// constexpr int PREFETCH_K = 16;

// // aligned allocation helper (64-byte)
// static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
//     void* p = nullptr;
//     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
//     return p;
// }
// static inline void aligned_free(void* p) { free(p); }

// // AVX2 4x4 microkernel: C[MR x NR] += A[MR x klen] * B[klen x NR]
// // A: MR x klen, stride lda
// // B: klen x NR, stride ldb (we load rows of B)
// // C: MR x NR, stride ldc
// static inline void microkernel4x4_avx2(const double* A, const double* B, double* C,
//                                        int lda, int ldb, int ldc, int klen) {
//     // Load C rows (each load grabs NR contiguous doubles)
//     __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
//     __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
//     __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
//     __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);

//     for (int k = 0; k < klen; ++k) {
//         // Prefetch a future row of B to hide memory latency
//         if (k + PREFETCH_K < klen) {
//             const char* ptr = (const char*)&B[(k + PREFETCH_K) * ldb];
//             __builtin_prefetch(ptr, 0, 3);
//         }
//         __m256d b = _mm256_loadu_pd(&B[k * ldb]); // B[k, j..j+3]
//         __m256d a0 = _mm256_broadcast_sd(&A[0 * lda + k]);
//         __m256d a1 = _mm256_broadcast_sd(&A[1 * lda + k]);
//         __m256d a2 = _mm256_broadcast_sd(&A[2 * lda + k]);
//         __m256d a3 = _mm256_broadcast_sd(&A[3 * lda + k]);

//         // fused multiply-add (FMA)
//         c0 = _mm256_fmadd_pd(a0, b, c0);
//         c1 = _mm256_fmadd_pd(a1, b, c1);
//         c2 = _mm256_fmadd_pd(a2, b, c2);
//         c3 = _mm256_fmadd_pd(a3, b, c3);
//     }

//     _mm256_storeu_pd(&C[0*ldc], c0);
//     _mm256_storeu_pd(&C[1*ldc], c1);
//     _mm256_storeu_pd(&C[2*ldc], c2);
//     _mm256_storeu_pd(&C[3*ldc], c3);
// }

// // Blocked GEMM driver (AVX2-only)
// // A, B, C are row-major flat arrays with leading dimension = N
// static void gemm_blocked_avx2(const double* A, const double* B, double* C, int N) {
//     const int lda = N, ldb = N, ldc = N;

//     // Parallelization: collapse outer tile loops to create enough tasks
//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int jc = 0; jc < N; jc += NC) {
//         for (int ic = 0; ic < N; ic += MC) {
//             for (int pc = 0; pc < N; pc += KC) {
//                 int jMax = std::min(jc + NC, N);
//                 int iMax = std::min(ic + MC, N);
//                 int pMax = std::min(pc + KC, N);

//                 for (int i = ic; i < iMax; i += MR) {
//                     int i_rem = std::min(MR, iMax - i);
//                     for (int j = jc; j < jMax; j += NR) {
//                         int j_rem = std::min(NR, jMax - j);

//                         double* Cblock = &C[i * ldc + j];
//                         const double* Ablock = &A[i * lda + pc];
//                         const double* Bblock = &B[pc * ldb + j];
//                         int klen = pMax - pc;

//                         // Full 4x4 tile -> AVX2 microkernel
//                         if (i_rem == MR && j_rem == NR) {
//                             microkernel4x4_avx2(Ablock, Bblock, Cblock, lda, ldb, ldc, klen);
//                         } else {
//                             // scalar fallback for tails (edges)
//                             for (int ii = 0; ii < i_rem; ++ii) {
//                                 for (int kk = 0; kk < klen; ++kk) {
//                                     double a = Ablock[ii * lda + kk];
//                                     for (int jj = 0; jj < j_rem; ++jj) {
//                                         Cblock[ii * ldc + jj] += a * Bblock[kk * ldb + jj];
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// int main(int argc, char** argv) {
//     if (argc < 3) {
//         cerr << "Usage: " << argv[0] << " N num_threads\n";
//         return 1;
//     }
//     int N = atoi(argv[1]);
//     int T = atoi(argv[2]);
//     if (N <= 0 || T <= 0) {
//         cerr << "N and num_threads must be > 0\n";
//         return 1;
//     }

//     // OpenMP settings
//     omp_set_dynamic(0);
//     omp_set_num_threads(T);

//     size_t nelems = (size_t)N * (size_t)N;
//     size_t bytes = nelems * sizeof(double);

//     double* A = (double*)aligned_malloc(bytes);
//     double* B = (double*)aligned_malloc(bytes);
//     double* C = (double*)aligned_malloc(bytes);
//     if (!A || !B || !C) {
//         cerr << "Allocation failed; try smaller N or increase RAM\n";
//         aligned_free(A); aligned_free(B); aligned_free(C);
//         return 1;
//     }

//     // Parallel first-touch initialization (deterministic)
//     #pragma omp parallel for schedule(static)
//     for (size_t idx = 0; idx < nelems; ++idx) {
//         // deterministic simple PRNG-ish init (fast + reproducible)
//         uint64_t v = 11400714819323198485ull * (idx + 1);
//         double x = (double)((v >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
//         A[idx] = x * 2.0 - 1.0;
//         uint64_t w = v ^ 0x9e3779b97f4a7c15ULL;
//         double y = (double)((w >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
//         B[idx] = y * 2.0 - 1.0;
//         C[idx] = 0.0;
//     }

//     // Warm-up message
//     cout << "Running AVX2 GEMM: N=" << N << " threads=" << T
//          << " MC=" << MC << " NC=" << NC << " KC=" << KC << " MR=" << MR << " NR=" << NR << "\n";

//     double t0 = omp_get_wtime();
//     gemm_blocked_avx2(A, B, C, N);
//     double t1 = omp_get_wtime();
//     double elapsed = t1 - t0;

//     // checksum as long double (reduction)
//     long double checksum = 0.0L;
//     #pragma omp parallel for reduction(+ : checksum)
//     for (size_t i = 0; i < nelems; ++i) checksum += C[i];

//     double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

//     cout << fixed << setprecision(6);
//     cout << "elapsed=" << elapsed << " s  GFLOPs=" << gflops
//          << "  checksum=" << setprecision(17) << (double)checksum << "\n";

//     aligned_free(A); aligned_free(B); aligned_free(C);
//     return 0;
// }


#include <bits/stdc++.h>
#include <omp.h>
#include <immintrin.h>
using namespace std;

// Blocking params (good defaults; tune for your CPU)
constexpr int MC = 256;   // L2-friendly rows block
constexpr int NC = 512;   // L3-friendly cols block
constexpr int KC = 128;   // L1-friendly depth block
constexpr int MR = 4;     // micro-kernel rows (AVX2)
constexpr int NR = 4;     // micro-kernel cols (AVX2)

// Prefetch depth (tuneable)
constexpr int PREFETCH_K = 64;

// aligned allocation helper (64-byte)
static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
}
static inline void aligned_free(void* p) { free(p); }

// AVX2 4x4 microkernel: C[MR x NR] += A[MR x klen] * B[klen x NR]
// A: MR x klen, stride lda
// B: klen x NR, stride ldb (we load rows of B)
// C: MR x NR, stride ldc
static inline void microkernel4x4_avx2(const double* A, const double* B, double* C,
                                       int lda, int ldb, int ldc, int klen) {
    // Load C rows (each load grabs NR contiguous doubles)
    __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
    __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
    __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
    __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);

    for (int k = 0; k < klen; ++k) {
        // Prefetch a future row of B to hide memory latency
        if (k + PREFETCH_K < klen) {
            const char* ptr = (const char*)&B[(k + PREFETCH_K) * ldb];
            __builtin_prefetch(ptr, 0, 3);
        }
        __m256d b = _mm256_loadu_pd(&B[k * ldb]); // B[k, j..j+3]
        __m256d a0 = _mm256_broadcast_sd(&A[0 * lda + k]);
        __m256d a1 = _mm256_broadcast_sd(&A[1 * lda + k]);
        __m256d a2 = _mm256_broadcast_sd(&A[2 * lda + k]);
        __m256d a3 = _mm256_broadcast_sd(&A[3 * lda + k]);

        // fused multiply-add (FMA)
        c0 = _mm256_fmadd_pd(a0, b, c0);
        c1 = _mm256_fmadd_pd(a1, b, c1);
        c2 = _mm256_fmadd_pd(a2, b, c2);
        c3 = _mm256_fmadd_pd(a3, b, c3);
    }

    _mm256_storeu_pd(&C[0*ldc], c0);
    _mm256_storeu_pd(&C[1*ldc], c1);
    _mm256_storeu_pd(&C[2*ldc], c2);
    _mm256_storeu_pd(&C[3*ldc], c3);
}

// Blocked GEMM driver (AVX2-only)
// A, B, C are row-major flat arrays with leading dimension = N
static void gemm_blocked_avx2(const double* A, const double* B, double* C, int N) {
    const int lda = N, ldb = N, ldc = N;

    // Parallelization: collapse outer tile loops to create enough tasks
    #pragma omp parallel for collapse(2) schedule(static)
    for (int jc = 0; jc < N; jc += NC) {
        for (int ic = 0; ic < N; ic += MC) {
            for (int pc = 0; pc < N; pc += KC) {
                int jMax = std::min(jc + NC, N);
                int iMax = std::min(ic + MC, N);
                int pMax = std::min(pc + KC, N);

                for (int i = ic; i < iMax; i += MR) {
                    int i_rem = std::min(MR, iMax - i);
                    for (int j = jc; j < jMax; j += NR) {
                        int j_rem = std::min(NR, jMax - j);

                        double* Cblock = &C[i * ldc + j];
                        const double* Ablock = &A[i * lda + pc];
                        const double* Bblock = &B[pc * ldb + j];
                        int klen = pMax - pc;

                        // Full 4x4 tile -> AVX2 microkernel
                        if (i_rem == MR && j_rem == NR) {
                            microkernel4x4_avx2(Ablock, Bblock, Cblock, lda, ldb, ldc, klen);
                        } else {
                            // scalar fallback for tails (edges)
                            for (int ii = 0; ii < i_rem; ++ii) {
                                for (int kk = 0; kk < klen; ++kk) {
                                    double a = Ablock[ii * lda + kk];
                                    for (int jj = 0; jj < j_rem; ++jj) {
                                        Cblock[ii * ldc + jj] += a * Bblock[kk * ldb + jj];
                                    }
                                }
                            }
                        }
                    }
                }
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
    if (N <= 0 || T <= 0) {
        cerr << "N and num_threads must be > 0\n";
        return 1;
    }

    // OpenMP settings
    omp_set_dynamic(0);
    omp_set_num_threads(T);

    size_t nelems = (size_t)N * (size_t)N;
    size_t bytes = nelems * sizeof(double);

    double* A = (double*)aligned_malloc(bytes);
    double* B = (double*)aligned_malloc(bytes);
    double* C = (double*)aligned_malloc(bytes);
    if (!A || !B || !C) {
        cerr << "Allocation failed; try smaller N or increase RAM\n";
        aligned_free(A); aligned_free(B); aligned_free(C);
        return 1;
    }

    // Parallel first-touch initialization (deterministic)
   
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < (size_t)N*N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // Warm-up message
    cout << "Running AVX2 GEMM: N=" << N << " threads=" << T
         << " MC=" << MC << " NC=" << NC << " KC=" << KC << " MR=" << MR << " NR=" << NR << "\n";

    double t0 = omp_get_wtime();
    gemm_blocked_avx2(A, B, C, N);
    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;

    // checksum as long double (reduction)
    long double checksum = 0.0L;
    #pragma omp parallel for reduction(+ : checksum)
    for (size_t i = 0; i < nelems; ++i) checksum += C[i];

    double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

    cout << fixed << setprecision(6);
    cout << "elapsed=" << elapsed << " s  GFLOPs=" << gflops
         << "  checksum=" << setprecision(2) << (double)checksum << "\n";

    aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}
