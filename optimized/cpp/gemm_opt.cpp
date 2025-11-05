
// // // // /*
// // // // C++ optimized skeleton for GEMM. Students should fill in optimizations:
// // // // - OpenMP parallelization
// // // // - Blocking / tiling
// // // // - SIMD intrinsics (AVX2 / AVX512)
// // // // - NUMA-aware allocations

// // // // Usage: ./gemm_opt N num_threads
// // // // */
// // // // #include <bits/stdc++.h>
// // // // #include <omp.h>

// // // // using namespace std;

// // // // static void matmul_naive(const double* A, const double* B, double* C, int N) {
// // // //     // simple triple-loop (row-major assumed)
// // // //     for (int i = 0; i < N; ++i) {
// // // //         for (int k = 0; k < N; ++k) {
// // // //             double a = A[i*N + k];
// // // //             for (int j = 0; j < N; ++j) {
// // // //                 C[i*N + j] += a * B[k*N + j];
// // // //             }
// // // //         }
// // // //     }
// // // // }

// // // // int main(int argc, char** argv) {
// // // //     if (argc < 3) {
// // // //         cerr << "Usage: " << argv[0] << " N num_threads\n";
// // // //         return 1;
// // // //     }
// // // //     int N = atoi(argv[1]);
// // // //     int T = atoi(argv[2]);
// // // //     omp_set_num_threads(T);

// // // //     vector<double> A(N*N), B(N*N), C(N*N);
// // // //     // reproducible RNG
// // // //     std::mt19937_64 rng(12345);
// // // //     std::normal_distribution<double> dist(0.0, 1.0);
// // // //     for (int i=0;i<N*N;i++) { A[i] = dist(rng); B[i] = dist(rng); C[i]=0.0; }

// // // //     double t0 = omp_get_wtime();
// // // //     matmul_naive(A.data(), B.data(), C.data(), N);
// // // //     double t1 = omp_get_wtime();
// // // //     cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

// // // //     // simple checksum to validate
// // // //     double s = 0; for (int i=0;i<N*N;i++) s += C[i];
// // // //     cout << "checksum=" << s << "\n";
// // // //     return 0;
// // // // }

// // // // // gemm_microkernel.cpp
// // // // // Usage: ./gemm_microkernel N num_threads
// // // // // Build: g++ -Ofast -march=native -mfma -fopenmp gemm_microkernel.cpp -o gemm_microkernel

// // // // gemm_avx2.cpp
// // // // AVX2-only optimized blocked GEMM tuned for N up to ~10000
// // // // - Compile with: g++ -O3 -mavx2 -mfma -fopenmp -o gemm_avx2 gemm_avx2.cpp
// // // // - Run: ./gemm_avx2 N num_threads
// // // // Notes: MR=4 NR=4 AVX2 micro-kernel; KC/MC/NC are tunable blocking parameters.
// // // // #include <bits/stdc++.h>
// // // // #include <omp.h>
// // // // #include <immintrin.h>
// // // // using namespace std;

// // // // // Micro-kernel params (fixed for AVX2)
// // // // constexpr int MR = 10; // micro-kernel rows
// // // // constexpr int NR = 8; // micro-kernel cols

// // // // // Prefetch depth (tuneable)
// // // // constexpr int PREFETCH_K = 16;

// // // // // aligned allocation helper (64-byte)
// // // // static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
// // // //     void* p = nullptr;
// // // //     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
// // // //     return p;
// // // // }
// // // // static inline void aligned_free(void* p) { free(p); }

// // // // // AVX2 4x4 microkernel
// // // // static inline void microkernel4x4_avx2(const double* A, const double* B, double* C,
// // // //                                        int lda, int ldb, int ldc, int klen) {
// // // //     __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
// // // //     __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
// // // //     __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
// // // //     __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);

// // // //     for (int k = 0; k < klen; ++k) {
// // // //         if (k + PREFETCH_K < klen) {
// // // //             const char* ptr = (const char*)&B[(k + PREFETCH_K) * ldb];
// // // //             __builtin_prefetch(ptr, 0, 3);
// // // //         }
// // // //         __m256d b = _mm256_loadu_pd(&B[k * ldb]);
// // // //         __m256d a0 = _mm256_broadcast_sd(&A[0 * lda + k]);
// // // //         __m256d a1 = _mm256_broadcast_sd(&A[1 * lda + k]);
// // // //         __m256d a2 = _mm256_broadcast_sd(&A[2 * lda + k]);
// // // //         __m256d a3 = _mm256_broadcast_sd(&A[3 * lda + k]);

// // // //         c0 = _mm256_fmadd_pd(a0, b, c0);
// // // //         c1 = _mm256_fmadd_pd(a1, b, c1);
// // // //         c2 = _mm256_fmadd_pd(a2, b, c2);
// // // //         c3 = _mm256_fmadd_pd(a3, b, c3);
// // // //     }

// // // //     _mm256_storeu_pd(&C[0*ldc], c0);
// // // //     _mm256_storeu_pd(&C[1*ldc], c1);
// // // //     _mm256_storeu_pd(&C[2*ldc], c2);
// // // //     _mm256_storeu_pd(&C[3*ldc], c3);
// // // // }

// // // // // Blocked GEMM driver (AVX2-only, runtime params)
// // // // static void gemm_blocked_avx2(const double* A, const double* B, double* C, int N,
// // // //                               int MC, int NC, int KC) {
// // // //     const int lda = N, ldb = N, ldc = N;

// // // //     #pragma omp parallel for collapse(2) schedule(static)
// // // //     for (int jc = 0; jc < N; jc += NC) {
// // // //         for (int ic = 0; ic < N; ic += MC) {
// // // //             for (int pc = 0; pc < N; pc += KC) {
// // // //                 int jMax = std::min(jc + NC, N);
// // // //                 int iMax = std::min(ic + MC, N);
// // // //                 int pMax = std::min(pc + KC, N);

// // // //                 for (int i = ic; i < iMax; i += MR) {
// // // //                     int i_rem = std::min(MR, iMax - i);
// // // //                     for (int j = jc; j < jMax; j += NR) {
// // // //                         int j_rem = std::min(NR, jMax - j);

// // // //                         double* Cblock = &C[i * ldc + j];
// // // //                         const double* Ablock = &A[i * lda + pc];
// // // //                         const double* Bblock = &B[pc * ldb + j];
// // // //                         int klen = pMax - pc;

// // // //                         if (i_rem == MR && j_rem == NR) {
// // // //                             microkernel4x4_avx2(Ablock, Bblock, Cblock, lda, ldb, ldc, klen);
// // // //                         } else {
// // // //                             for (int ii = 0; ii < i_rem; ++ii) {
// // // //                                 for (int kk = 0; kk < klen; ++kk) {
// // // //                                     double a = Ablock[ii * lda + kk];
// // // //                                     for (int jj = 0; jj < j_rem; ++jj) {
// // // //                                         Cblock[ii * ldc + jj] += a * Bblock[kk * ldb + jj];
// // // //                                     }
// // // //                                 }
// // // //                             }
// // // //                         }
// // // //                     }
// // // //                 }
// // // //             }
// // // //         }
// // // //     }
// // // // }

// // // // int main(int argc, char** argv) {
// // // //     if (argc < 3) {
// // // //         cerr << "Usage: " << argv[0] << " N num_threads\n";
// // // //         return 1;
// // // //     }
// // // //     int N = atoi(argv[1]);
// // // //     int T = atoi(argv[2]);
// // // //     if (N <= 0 || T <= 0) {
// // // //         cerr << "N and num_threads must be > 0\n";
// // // //         return 1;
// // // //     }

// // // //     // OpenMP settings
// // // //     omp_set_dynamic(0);
// // // //     omp_set_num_threads(T);

// // // //     size_t nelems = (size_t)N * (size_t)N;
// // // //     size_t bytes = nelems * sizeof(double);

// // // //     double* A = (double*)aligned_malloc(bytes);
// // // //     double* B = (double*)aligned_malloc(bytes);
// // // //     double* C = (double*)aligned_malloc(bytes);
// // // //     if (!A || !B || !C) {
// // // //         cerr << "Allocation failed; try smaller N or increase RAM\n";
// // // //         aligned_free(A); aligned_free(B); aligned_free(C);
// // // //         return 1;
// // // //     }

// // // //     // Init once (same A, B for all runs)
// // // //     #pragma omp parallel for schedule(static)
// // // //     for (size_t idx = 0; idx < nelems; ++idx) {
// // // //         uint64_t v = 11400714819323198485ull * (idx + 1);
// // // //         double x = (double)((v >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
// // // //         A[idx] = x * 2.0 - 1.0;
// // // //         uint64_t w = v ^ 0x9e3779b97f4a7c15ULL;
// // // //         double y = (double)((w >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
// // // //         B[idx] = y * 2.0 - 1.0;
// // // //         C[idx] = 0.0;
// // // //     }
// // // // // 64 , 256
// // // //     vector<int> KC_list = {128};
// // // //     // 512 128
// // // //     vector<int> MC_list = {256};
// // // //     // 1024 2048
// // // //     vector<int> NC_list = {512};

// // // //     // Best-tracker
// // // //     double best_gflops = -1.0;
// // // //     double best_elapsed = std::numeric_limits<double>::infinity();
// // // //     int best_KC = 0, best_MC = 0, best_NC = 0;

// // // //     cout << fixed << setprecision(6);
// // // //     cout << "config_csv,KC,MC,NC,N,threads,elapsed_s,GFLOPs,checksum\n";

// // // //     for (int KC : KC_list) {
// // // //         for (int MC : MC_list) {
// // // //             for (int NC : NC_list) {

// // // //                 // fresh C for this run
// // // //                 #pragma omp parallel for schedule(static)
// // // //                 for (size_t i = 0; i < nelems; ++i) C[i] = 0.0;

// // // //                 // banner
// // // //                 cout << "run,"
// // // //                      << KC << "," << MC << "," << NC << ","
// // // //                      << N << "," << T << ",";

// // // //                 double t0 = omp_get_wtime();
// // // //                 gemm_blocked_avx2(A, B, C, N, MC, NC, KC);
// // // //                 double t1 = omp_get_wtime();
// // // //                 double elapsed = max(1e-12, t1 - t0); // guard divide-by-zero

// // // //                 long double checksum = 0.0L;
// // // //                 #pragma omp parallel for reduction(+ : checksum)
// // // //                 for (size_t i = 0; i < nelems; ++i) checksum += C[i];

// // // //                 double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

// // // //                 cout << elapsed << "," << gflops << ","
// // // //                      << setprecision(17) << (double)checksum << "\n";
// // // //                 cout << setprecision(6); // restore for later prints

// // // //                 // update best (primary: max GFLOPs, tie-break: min elapsed)
// // // //                 if (gflops > best_gflops || (fabs(gflops - best_gflops) < 1e-9 && elapsed < best_elapsed)) {
// // // //                     best_gflops = gflops;
// // // //                     best_elapsed = elapsed;
// // // //                     best_KC = KC; best_MC = MC; best_NC = NC;
// // // //                 }
// // // //             }
// // // //         }
// // // //     }

// // // //     // Final best summary
// // // //     cout << "\n=== BEST CONFIG ===\n"
// // // //          << "KC=" << best_KC << "  MC=" << best_MC << "  NC=" << best_NC
// // // //          << "  -> GFLOPs=" << best_gflops
// // // //          << "  elapsed=" << best_elapsed << "s"<<"    MR  "<< MR<<"   NR  "<<NR<<"\n";

// // // //     aligned_free(A); aligned_free(B); aligned_free(C);
// // // //     return 0;
// // // // }

// // // #include <bits/stdc++.h>
// // // #include <omp.h>
// // // #include <immintrin.h>
// // // using namespace std;

// // // // Micro-kernel params (fixed for AVX2)
// // // // ⚠️ Your kernel is 4x4; using MR=10,NR=8 will rarely hit the fast path.
// // // // For best performance set MR=4, NR=4.
// // // constexpr int MR = 10; // micro-kernel rows
// // // constexpr int NR = 8;  // micro-kernel cols

// // // // aligned allocation helper (64-byte)
// // // static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
// // //     void* p = nullptr;
// // //     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
// // //     return p;
// // // }
// // // static inline void aligned_free(void* p) { free(p); }

// // // // AVX2 4x4 microkernel with runtime prefetch distance
// // // static inline void microkernel4x4_avx2(const double* A, const double* B, double* C,
// // //                                        int lda, int ldb, int ldc, int klen, int prefetchK) {
// // //     __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
// // //     __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
// // //     __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
// // //     __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);

// // //     for (int k = 0; k < klen; ++k) {
// // //         if (prefetchK > 0 && k + prefetchK < klen) {
// // //             const char* ptr = (const char*)&B[(k + prefetchK) * ldb];
// // //             __builtin_prefetch(ptr, 0, 3);
// // //         }
// // //         __m256d b  = _mm256_loadu_pd(&B[k * ldb]);        // B[k, j..j+3]
// // //         __m256d a0 = _mm256_broadcast_sd(&A[0 * lda + k]); // A[i+0, pc+k]
// // //         __m256d a1 = _mm256_broadcast_sd(&A[1 * lda + k]);
// // //         __m256d a2 = _mm256_broadcast_sd(&A[2 * lda + k]);
// // //         __m256d a3 = _mm256_broadcast_sd(&A[3 * lda + k]);

// // //         c0 = _mm256_fmadd_pd(a0, b, c0);
// // //         c1 = _mm256_fmadd_pd(a1, b, c1);
// // //         c2 = _mm256_fmadd_pd(a2, b, c2);
// // //         c3 = _mm256_fmadd_pd(a3, b, c3);
// // //     }

// // //     _mm256_storeu_pd(&C[0*ldc], c0);
// // //     _mm256_storeu_pd(&C[1*ldc], c1);
// // //     _mm256_storeu_pd(&C[2*ldc], c2);
// // //     _mm256_storeu_pd(&C[3*ldc], c3);
// // // }

// // // // Blocked GEMM driver (passes runtime prefetchK)
// // // static void gemm_blocked_avx2(const double* A, const double* B, double* C, int N,
// // //                               int MC, int NC, int KC, int prefetchK) {
// // //     const int lda = N, ldb = N, ldc = N;

// // //     #pragma omp parallel for collapse(2) schedule(static)
// // //     for (int jc = 0; jc < N; jc += NC) {
// // //         for (int ic = 0; ic < N; ic += MC) {
// // //             for (int pc = 0; pc < N; pc += KC) {
// // //                 int jMax = std::min(jc + NC, N);
// // //                 int iMax = std::min(ic + MC, N);
// // //                 int pMax = std::min(pc + KC, N);

// // //                 for (int i = ic; i < iMax; i += MR) {
// // //                     int i_rem = std::min(MR, iMax - i);
// // //                     for (int j = jc; j < jMax; j += NR) {
// // //                         int j_rem = std::min(NR, jMax - j);

// // //                         double* Cblock        = &C[i * ldc + j];
// // //                         const double* Ablock  = &A[i * lda + pc];
// // //                         const double* Bblock  = &B[pc * ldb + j];
// // //                         int klen = pMax - pc;

// // //                         if (i_rem == MR && j_rem == NR) {
// // //                             microkernel4x4_avx2(Ablock, Bblock, Cblock, lda, ldb, ldc, klen, prefetchK);
// // //                         } else {
// // //                             // scalar remainder for borders
// // //                             for (int ii = 0; ii < i_rem; ++ii) {
// // //                                 for (int kk = 0; kk < klen; ++kk) {
// // //                                     double a = Ablock[ii * lda + kk];
// // //                                     for (int jj = 0; jj < j_rem; ++jj) {
// // //                                         Cblock[ii * ldc + jj] += a * Bblock[kk * ldb + jj];
// // //                                     }
// // //                                 }
// // //                             }
// // //                         }
// // //                     }
// // //                 }
// // //             }
// // //         }
// // //     }
// // // }

// // // int main(int argc, char** argv) {
// // //     if (argc < 3) {
// // //         cerr << "Usage: " << argv[0] << " N num_threads\n";
// // //         return 1;
// // //     }
// // //     int N = atoi(argv[1]);
// // //     int T = atoi(argv[2]);
// // //     if (N <= 0 || T <= 0) {
// // //         cerr << "N and num_threads must be > 0\n";
// // //         return 1;
// // //     }

// // //     // OpenMP settings
// // //     omp_set_dynamic(0);
// // //     omp_set_num_threads(T);

// // //     size_t nelems = (size_t)N * (size_t)N;
// // //     size_t bytes  = nelems * sizeof(double);

// // //     double* A = (double*)aligned_malloc(bytes);
// // //     double* B = (double*)aligned_malloc(bytes);
// // //     double* C = (double*)aligned_malloc(bytes);
// // //     if (!A || !B || !C) {
// // //         cerr << "Allocation failed; try smaller N or increase RAM\n";
// // //         aligned_free(A); aligned_free(B); aligned_free(C);
// // //         return 1;
// // //     }

// // //     // Deterministic init
// // //     #pragma omp parallel for schedule(static)
// // //     for (size_t idx = 0; idx < nelems; ++idx) {
// // //         uint64_t v = 11400714819323198485ull * (idx + 1);
// // //         double x = (double)((v >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
// // //         A[idx] = x * 2.0 - 1.0;
// // //         uint64_t w = v ^ 0x9e3779b97f4a7c15ULL;
// // //         double y = (double)((w >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
// // //         B[idx] = y * 2.0 - 1.0;
// // //         C[idx] = 0.0;
// // //     }

// // //     // Tunable sets
// // //     vector<int> KC_list = {128};
// // //     vector<int> MC_list = {256};
// // //     vector<int> NC_list = {512};
// // //     vector<int> prefetch_list = {48}; // <— sweep prefetch distance

// // //     // Best-tracker
// // //     double best_gflops = -1.0;
// // //     double best_elapsed = std::numeric_limits<double>::infinity();
// // //     int best_KC = 0, best_MC = 0, best_NC = 0, best_prefetch = 0;

// // //     cout << fixed << setprecision(6);
// // //     cout << "config_csv,KC,MC,NC,N,threads,prefetchK,elapsed_s,GFLOPs,checksum\n";

// // //     for (int PREF : prefetch_list) {
// // //         for (int KC : KC_list) {
// // //             for (int MC : MC_list) {
// // //                 for (int NC : NC_list) {

// // //                     // fresh C for this run
// // //                     #pragma omp parallel for schedule(static)
// // //                     for (size_t i = 0; i < nelems; ++i) C[i] = 0.0;

// // //                     // banner
// // //                     cout << "run," << KC << "," << MC << "," << NC << ","
// // //                          << N << "," << T << "," << PREF << ",";

// // //                     double t0 = omp_get_wtime();
// // //                     gemm_blocked_avx2(A, B, C, N, MC, NC, KC, PREF);
// // //                     double t1 = omp_get_wtime();
// // //                     double elapsed = max(1e-12, t1 - t0);

// // //                     long double checksum = 0.0L;
// // //                     #pragma omp parallel for reduction(+ : checksum) schedule(static)
// // //                     for (size_t i = 0; i < nelems; ++i) checksum += C[i];

// // //                     double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

// // //                     cout << elapsed << "," << gflops << ","
// // //                          << setprecision(17) << (double)checksum << "\n";
// // //                     cout << setprecision(6);

// // //                     if (gflops > best_gflops || (fabs(gflops - best_gflops) < 1e-9 && elapsed < best_elapsed)) {
// // //                         best_gflops = gflops;
// // //                         best_elapsed = elapsed;
// // //                         best_KC = KC; best_MC = MC; best_NC = NC;
// // //                         best_prefetch = PREF;
// // //                     }
// // //                 }
// // //             }
// // //         }
// // //     }

// // //     cout << "\n=== BEST CONFIG ===\n"
// // //          << "KC=" << best_KC << "  MC=" << best_MC << "  NC=" << best_NC
// // //          << "  prefetchK=" << best_prefetch
// // //          << "  -> GFLOPs=" << best_gflops
// // //          << "  elapsed=" << best_elapsed << "s"
// // //          << "    MR " << MR << "   NR " << NR << "\n";

// // //     aligned_free(A); aligned_free(B); aligned_free(C);
// // //     return 0;
// // // }


// // int **H = (int**) malloc((len1 + 1) * sizeof(int *));
// // ...
// // H[i] = (int*) calloc((len2 + 1), sizeof(int));
// // ...
// // char *seq1 = (char*) malloc((N + 1) * sizeof(char));
// // char *seq2 = (char*) malloc((N + 1) * sizeof(char));




// // #include <stdio.h>
// // #include <stdlib.h>
// // #include <string.h>
// // #include <time.h>

// // #define MAX(A,B) ((A) > (B) ? (A) : (B))
// // #define MATCH     2
// // #define MISMATCH -1
// // #define GAP      -2

// // void generate_sequence(char *seq, int n) {
// //     const char alphabet[] = "ACGT";
// //     for (int i = 0; i < n; i++)
// //         seq[i] = alphabet[rand() % 4];
// //     seq[n] = '\0';
// // }

// // int smith_waterman(const char *seq1, const char *seq2, int len1, int len2) {
// //     int **H = (int**) malloc((len1 + 1) * sizeof(int *));
// //     for (int i = 0; i <= len1; i++)
// //         H[i] = (int*)calloc(len2 + 1, sizeof(int));

// //     int max_score = 0;
// //     for (int i = 1; i <= len1; i++) {
// //         for (int j = 1; j <= len2; j++) {
// //             int match = H[i - 1][j - 1] + (seq1[i - 1] == seq2[j - 1] ? MATCH : MISMATCH);
// //             int del = H[i - 1][j] + GAP;
// //             int ins = H[i][j - 1] + GAP;
// //             H[i][j] = MAX(0, MAX(match, MAX(del, ins)));
// //             if (H[i][j] > max_score)
// //                 max_score = H[i][j];
// //         }
// //     }

// //     for (int i = 0; i <= len1; i++) free(H[i]);
// //     free(H);

// //     return max_score;
// // }

// // int main(int argc, char **argv) {
// //     if (argc < 2) {
// //         fprintf(stderr, "Usage: %s <sequence_length>\n", argv[0]);
// //         return 1;
// //     }

// //     int N = atoi(argv[1]);
// //     srand(42);

// //     char *seq1 = (char*)malloc((N + 1) * sizeof(char));
// //     char *seq2 = (char*)malloc((N + 1) * sizeof(char));

// //     generate_sequence(seq1, N);
// //     generate_sequence(seq2, N);

// //     clock_t start = clock();
// //     int score = smith_waterman(seq1, seq2, N, N);
// //     clock_t end = clock();

// //     double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
// //     printf("Sequence length: %d\n", N);
// //     printf("Smith-Waterman score: %d\n", score);
// //     printf("Execution time: %.6f seconds\n", elapsed);

// //     free(seq1);
// //     free(seq2);
// //     return 0;
// // }


// #include <bits/stdc++.h>
// #include <omp.h>
// #include <immintrin.h>
// using namespace std;

// // Micro-kernel params (match AVX2: 4 doubles per 256-bit register)
// constexpr int MR = 10; // micro-kernel rows
// constexpr int NR = 8; // micro-kernel cols

// // aligned allocation helper (64-byte)
// static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
//     void* p = nullptr;
//     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
//     return p;
// }
// static inline void aligned_free(void* p) { free(p); }

// // AVX2 4x4 microkernel with runtime prefetch distance
// static inline void microkernel4x4_avx2(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
//                                        int lda, int ldb, int ldc, int klen, int prefetchK) {
//     // load C 4 elements per row
//     __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
//     __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
//     __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
//     __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);

//     int k = 0;
//     // simple unroll-by-2 for ILP
//     for (; k + 1 < klen; k += 2) {
//         if (prefetchK > 0 && k + prefetchK < klen) {
//             const char* ptr = (const char*)&B[(k + prefetchK) * ldb];
//             __builtin_prefetch(ptr, 0, 3);
//         }
//         __m256d b0  = _mm256_loadu_pd(&B[(k) * ldb]);
//         __m256d a00 = _mm256_broadcast_sd(&A[0 * lda + k]);
//         __m256d a10 = _mm256_broadcast_sd(&A[1 * lda + k]);
//         __m256d a20 = _mm256_broadcast_sd(&A[2 * lda + k]);
//         __m256d a30 = _mm256_broadcast_sd(&A[3 * lda + k]);
//         c0 = _mm256_fmadd_pd(a00, b0, c0);
//         c1 = _mm256_fmadd_pd(a10, b0, c1);
//         c2 = _mm256_fmadd_pd(a20, b0, c2);
//         c3 = _mm256_fmadd_pd(a30, b0, c3);

//         if (prefetchK > 0 && k + 1 + prefetchK < klen) {
//             const char* ptr2 = (const char*)&B[(k + 1 + prefetchK) * ldb];
//             __builtin_prefetch(ptr2, 0, 3);
//         }
//         __m256d b1  = _mm256_loadu_pd(&B[(k+1) * ldb]);
//         __m256d a01 = _mm256_broadcast_sd(&A[0 * lda + k+1]);
//         __m256d a11 = _mm256_broadcast_sd(&A[1 * lda + k+1]);
//         __m256d a21 = _mm256_broadcast_sd(&A[2 * lda + k+1]);
//         __m256d a31 = _mm256_broadcast_sd(&A[3 * lda + k+1]);
//         c0 = _mm256_fmadd_pd(a01, b1, c0);
//         c1 = _mm256_fmadd_pd(a11, b1, c1);
//         c2 = _mm256_fmadd_pd(a21, b1, c2);
//         c3 = _mm256_fmadd_pd(a31, b1, c3);
//     }
//     for (; k < klen; ++k) {
//         if (prefetchK > 0 && k + prefetchK < klen) {
//             const char* ptr = (const char*)&B[(k + prefetchK) * ldb];
//             __builtin_prefetch(ptr, 0, 3);
//         }
//         __m256d b  = _mm256_loadu_pd(&B[k * ldb]);
//         __m256d a0 = _mm256_broadcast_sd(&A[0 * lda + k]);
//         __m256d a1 = _mm256_broadcast_sd(&A[1 * lda + k]);
//         __m256d a2 = _mm256_broadcast_sd(&A[2 * lda + k]);
//         __m256d a3 = _mm256_broadcast_sd(&A[3 * lda + k]);
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

// // Blocked GEMM driver (passes runtime prefetchK) - now accepts strides
// static void gemm_blocked_avx2(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
//                               int N, int MC, int NC, int KC, int prefetchK, int lda, int ldb, int ldc) {

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

//                         double* Cblock        = &C[i * ldc + j];
//                         const double* Ablock  = &A[i * lda + pc];
//                         const double* Bblock  = &B[pc * ldb + j];
//                         int klen = pMax - pc;

//                         if (__builtin_expect(i_rem == MR && j_rem == NR, 1)) {
//                             microkernel4x4_avx2(Ablock, Bblock, Cblock, lda, ldb, ldc, klen, prefetchK);
//                         } else {
//                             // scalar remainder for borders (rare path)
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
//         cerr << "Usage: " << argv[0] << " N num_threads [MC NC KC PREFETCH]" << endl;
//         return 1;
//     }
//     int N = atoi(argv[1]);
//     int T = atoi(argv[2]);
//     if (N <= 0 || T <= 0) {
//         cerr << "N and num_threads must be > 0" << endl;
//         return 1;
//     }

//     int MC = (argc > 3) ? atoi(argv[3]) : 256;
//     int NC = (argc > 4) ? atoi(argv[4]) : 512;
//     int KC = (argc > 5) ? atoi(argv[5]) : 128;
//     int PREF = (argc > 6) ? atoi(argv[6]) : 48;

//     // padding to avoid power-of-two aliasing (in doubles)
//     int pad = 16; // 16 doubles = 128 bytes; change to 8/32 if you want
//     int lda = N + pad;
//     int ldb = N + pad;
//     int ldc = N + pad;

//     // OpenMP
//     omp_set_dynamic(0);
//     omp_set_num_threads(T);

//     size_t nelemsA = (size_t)lda * (size_t)N;
//     size_t nelemsB = (size_t)ldb * (size_t)N;
//     size_t nelemsC = (size_t)ldc * (size_t)N;

//     double* A = (double*)aligned_malloc(sizeof(double) * nelemsA);
//     double* B = (double*)aligned_malloc(sizeof(double) * nelemsB);
//     double* C = (double*)aligned_malloc(sizeof(double) * nelemsC);
//     if (!A || !B || !C) {
//         cerr << "Allocation failed; try smaller N or increase RAM" << endl;
//         if (A) aligned_free(A);
//         if (B) aligned_free(B);
//         if (C) aligned_free(C);
//         return 1;
//     }

//     // Deterministic init using row-stride = lda/ldb
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < N; ++j) {
//             size_t idxA = (size_t)i * lda + (size_t)j;
//             size_t idxB = (size_t)i * ldb + (size_t)j;
//             uint64_t v = 11400714819323198485ull * (idxA + 1);
//             double x = (double)((v >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
//             A[idxA] = x * 2.0 - 1.0;
//             uint64_t w = v ^ 0x9e3779b97f4a7c15ULL;
//             double y = (double)((w >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
//             B[idxB] = y * 2.0 - 1.0;
//             C[(size_t)i * ldc + (size_t)j] = 0.0;
//         }
//     }

//     cout << fixed << setprecision(6);
//     cout << "config_csv,KC,MC,NC,N,threads,prefetchK,elapsed_s,GFLOPs,checksum\n";

//     // warm-up
//     gemm_blocked_avx2(A, B, C, N, MC, NC, KC, PREF, lda, ldb, ldc);

//     // fresh C
//     #pragma omp parallel for schedule(static)
//     for (size_t i = 0; i < (size_t)N; ++i) {
//         for (size_t j = 0; j < (size_t)N; ++j) C[i*ldc + j] = 0.0;
//     }

//     cout << "run," << KC << "," << MC << "," << NC << "," << N << "," << T << "," << PREF << ",";

//     double t0 = omp_get_wtime();
//     gemm_blocked_avx2(A, B, C, N, MC, NC, KC, PREF, lda, ldb, ldc);
//     double t1 = omp_get_wtime();
//     double elapsed = max(1e-12, t1 - t0);

//     long double checksum = 0.0L;
//     #pragma omp parallel for reduction(+ : checksum) schedule(static)
//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < N; ++j) checksum += C[(size_t)i * ldc + (size_t)j];
//     }

//     double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

//     cout << elapsed << "," << gflops << "," << setprecision(17) << (double)checksum << "\n";
//     cout << setprecision(6);

//     cout << "\n=== BEST CONFIG ===\n"
//          << "KC=" << KC << "  MC=" << MC << "  NC=" << NC
//          << "  prefetchK=" << PREF
//          << "  -> GFLOPs=" << gflops
//          << "  elapsed=" << elapsed << "s"
//          << "    MR " << MR << "   NR " << NR << "\n";

//     aligned_free(A); aligned_free(B); aligned_free(C);
//     return 0;
// }



#include <bits/stdc++.h>
#include <omp.h>
#include <immintrin.h>
using namespace std;

// Micro-kernel params (match AVX2: 4 doubles per 256-bit register)
constexpr int MR = 10; // micro-kernel rows
constexpr int NR = 8; // micro-kernel cols

// aligned allocation helper (64-byte)
static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
}
static inline void aligned_free(void* p) { free(p); }

// Micro-kernel: reads B from packed buffer (ldb_pack == NR), A and C use lda/ldc
static inline void microkernel4x4_avx2(const double* __restrict__ A, const double* __restrict__ Bpack, double* __restrict__ C,
                                       int lda, int ldb_pack, int ldc, int klen, int prefetchK) {
    // load C rows (4 elements each)
    __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
    __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
    __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
    __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);

    int k = 0;
    for (; k + 1 < klen; k += 2) {
        if (prefetchK > 0 && k + prefetchK < klen) {
            const char* ptr = (const char*)&Bpack[(k + prefetchK) * ldb_pack];
            __builtin_prefetch(ptr, 0, 3);
        }
        __m256d b0 = _mm256_loadu_pd(&Bpack[(k) * ldb_pack]);
        __m256d a00 = _mm256_broadcast_sd(&A[0 * lda + k]);
        __m256d a10 = _mm256_broadcast_sd(&A[1 * lda + k]);
        __m256d a20 = _mm256_broadcast_sd(&A[2 * lda + k]);
        __m256d a30 = _mm256_broadcast_sd(&A[3 * lda + k]);
        c0 = _mm256_fmadd_pd(a00, b0, c0);
        c1 = _mm256_fmadd_pd(a10, b0, c1);
        c2 = _mm256_fmadd_pd(a20, b0, c2);
        c3 = _mm256_fmadd_pd(a30, b0, c3);

        if (prefetchK > 0 && k + 1 + prefetchK < klen) {
            const char* ptr2 = (const char*)&Bpack[(k + 1 + prefetchK) * ldb_pack];
            __builtin_prefetch(ptr2, 0, 3);
        }
        __m256d b1 = _mm256_loadu_pd(&Bpack[(k + 1) * ldb_pack]);
        __m256d a01 = _mm256_broadcast_sd(&A[0 * lda + k+1]);
        __m256d a11 = _mm256_broadcast_sd(&A[1 * lda + k+1]);
        __m256d a21 = _mm256_broadcast_sd(&A[2 * lda + k+1]);
        __m256d a31 = _mm256_broadcast_sd(&A[3 * lda + k+1]);
        c0 = _mm256_fmadd_pd(a01, b1, c0);
        c1 = _mm256_fmadd_pd(a11, b1, c1);
        c2 = _mm256_fmadd_pd(a21, b1, c2);
        c3 = _mm256_fmadd_pd(a31, b1, c3);
    }
    for (; k < klen; ++k) {
        if (prefetchK > 0 && k + prefetchK < klen) {
            const char* ptr = (const char*)&Bpack[(k + prefetchK) * ldb_pack];
            __builtin_prefetch(ptr, 0, 3);
        }
        __m256d b = _mm256_loadu_pd(&Bpack[k * ldb_pack]);
        __m256d a0 = _mm256_broadcast_sd(&A[0 * lda + k]);
        __m256d a1 = _mm256_broadcast_sd(&A[1 * lda + k]);
        __m256d a2 = _mm256_broadcast_sd(&A[2 * lda + k]);
        __m256d a3 = _mm256_broadcast_sd(&A[3 * lda + k]);
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

// Blocked GEMM driver (passes runtime prefetchK) - with B packing and loops reordered
static void gemm_blocked_avx2(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                              int N, int MC, int NC, int KC, int prefetchK, int lda, int ldb, int ldc) {

    #pragma omp parallel for collapse(2) schedule(static)
    for (int jc = 0; jc < N; jc += NC) {
        for (int ic = 0; ic < N; ic += MC) {
            for (int pc = 0; pc < N; pc += KC) {
                int jMax = std::min(jc + NC, N);
                int iMax = std::min(ic + MC, N);
                int pMax = std::min(pc + KC, N);
                int klen = pMax - pc;

                // iterate over small j-blocks and pack B for each
                for (int j = jc; j < jMax; j += NR) {
                    int j_rem = std::min(NR, jMax - j);
                    // pack Bblock into contiguous Bpack[klen][NR]
                    std::vector<double> Bpack((size_t)klen * NR);
                    const double* Bblock = &B[pc * ldb + j];
                    for (int kk = 0; kk < klen; ++kk) {
                        for (int jj = 0; jj < j_rem; ++jj) {
                            Bpack[(size_t)kk * NR + jj] = Bblock[(size_t)kk * ldb + jj];
                        }
                        for (int jj = j_rem; jj < NR; ++jj) {
                            Bpack[(size_t)kk * NR + jj] = 0.0;
                        }
                    }

                    // now iterate over i-blocks and call microkernel
                    for (int i = ic; i < iMax; i += MR) {
                        int i_rem = std::min(MR, iMax - i);

                        double* Cblock       = &C[i * ldc + j];
                        const double* Ablock = &A[i * lda + pc];

                        if (__builtin_expect(i_rem == MR && j_rem == NR, 1)) {
                            microkernel4x4_avx2(Ablock, Bpack.data(), Cblock, lda, NR, ldc, klen, prefetchK);
                        } else {
                            // scalar remainder
                            for (int ii = 0; ii < i_rem; ++ii) {
                                for (int kk = 0; kk < klen; ++kk) {
                                    double a = Ablock[ii * lda + kk];
                                    for (int jj = 0; jj < j_rem; ++jj) {
                                        Cblock[ii * ldc + jj] += a * Bblock[kk * ldb + jj];
                                    }
                                }
                            }
                        }
                    } // i loop
                } // j-block
            } // pc
        } // ic
    } // jc
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads [MC NC KC PREFETCH]" << endl;
        return 1;
    }
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    if (N <= 0 || T <= 0) {
        cerr << "N and num_threads must be > 0" << endl;
        return 1;
    }

    int MC = (argc > 3) ? atoi(argv[3]) : 256;
    int NC = (argc > 4) ? atoi(argv[4]) : 512;
    int KC = (argc > 5) ? atoi(argv[5]) : 128;
    int PREF = (argc > 6) ? atoi(argv[6]) : 48;

    // padding to avoid power-of-two aliasing (in doubles)
    int pad = 16; // 16 doubles = 128 bytes; change to 8/32 if you want
    int lda = N + pad;
    int ldb = N + pad;
    int ldc = N + pad;

    // OpenMP
    omp_set_dynamic(0);
    omp_set_num_threads(T);

    size_t nelemsA = (size_t)lda * (size_t)N;
    size_t nelemsB = (size_t)ldb * (size_t)N;
    size_t nelemsC = (size_t)ldc * (size_t)N;

    double* A = (double*)aligned_malloc(sizeof(double) * nelemsA);
    double* B = (double*)aligned_malloc(sizeof(double) * nelemsB);
    double* C = (double*)aligned_malloc(sizeof(double) * nelemsC);
    if (!A || !B || !C) {
        cerr << "Allocation failed; try smaller N or increase RAM" << endl;
        if (A) aligned_free(A);
        if (B) aligned_free(B);
        if (C) aligned_free(C);
        return 1;
    }

    // Deterministic init using row-stride = lda/ldb
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            size_t idxA = (size_t)i * lda + (size_t)j;
            size_t idxB = (size_t)i * ldb + (size_t)j;
            uint64_t v = 11400714819323198485ull * (idxA + 1);
            double x = (double)((v >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
            A[idxA] = x * 2.0 - 1.0;
            uint64_t w = v ^ 0x9e3779b97f4a7c15ULL;
            double y = (double)((w >> 1) & 0x7fffffffffffffffULL) / (double)0x7fffffffffffffffULL;
            B[idxB] = y * 2.0 - 1.0;
            C[(size_t)i * ldc + (size_t)j] = 0.0;
        }
    }

    cout << fixed << setprecision(6);
    cout << "config_csv,KC,MC,NC,N,threads,prefetchK,elapsed_s,GFLOPs,checksum";

    // warm-up
    gemm_blocked_avx2(A, B, C, N, MC, NC, KC, PREF, lda, ldb, ldc);

    // fresh C
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)N; ++i) {
        for (size_t j = 0; j < (size_t)N; ++j) C[i*ldc + j] = 0.0;
    }

    cout << "run," << KC << "," << MC << "," << NC << "," << N << "," << T << "," << PREF << ",";

    double t0 = omp_get_wtime();
    gemm_blocked_avx2(A, B, C, N, MC, NC, KC, PREF, lda, ldb, ldc);
    double t1 = omp_get_wtime();
    double elapsed = max(1e-12, t1 - t0);

    long double checksum = 0.0L;
    #pragma omp parallel for reduction(+ : checksum) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) checksum += C[(size_t)i * ldc + (size_t)j];
    }

    double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

    cout << elapsed << "," << gflops << "," << setprecision(17) << (double)checksum << "";
    cout << setprecision(6);

    cout <<"\n" <<"=== BEST CONFIG ==="
         << "KC=" << KC << "  MC=" << MC << "  NC=" << NC
         << "  prefetchK=" << PREF
         << "  -> GFLOPs=" << gflops
         << "  elapsed=" << elapsed << "s"
         << "    MR " << MR << "   NR " << NR << "\n";

    aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}



