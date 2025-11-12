// // gemm_4x4_tiled.cpp
// // Simple tiled GEMM with a 4x4 AVX2 micro-kernel + scalar tail fallback
// // Compile: g++ -O3 -march=native -mavx2 -mfma -fopenmp gemm_4x4_tiled.cpp -o gemm_4x4_tiled

// #include <bits/stdc++.h>
// #include <immintrin.h>
// #include <omp.h>
// using namespace std;

// constexpr int MR = 4;
// constexpr int NR = 4;

// // aligned alloc
// static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
//     void* p = nullptr;
//     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
//     return p;
// }
// static inline void aligned_free(void* p) { free(p); }

// // 4x4 AVX2 micro-kernel
// // A_tile: pointer to A[i, k] (row-major) with lda stride
// // B_tile: pointer to B[k, j] (row-major) with ldb stride
// // C_tile: pointer to C[i, j] (row-major) with ldc stride
// // K: length of reduction dimension for this tile
// static inline void microkernel4x4(const double* __restrict__ A_tile,
//                                   const double* __restrict__ B_tile,
//                                   double* __restrict__ C_tile,
//                                   int lda, int ldb, int ldc,
//                                   int K)
// {
//     // load C rows (each load loads 4 consecutive doubles from the row start)
//     __m256d c0 = _mm256_loadu_pd(&C_tile[0 * ldc]);
//     __m256d c1 = _mm256_loadu_pd(&C_tile[1 * ldc]);
//     __m256d c2 = _mm256_loadu_pd(&C_tile[2 * ldc]);
//     __m256d c3 = _mm256_loadu_pd(&C_tile[3 * ldc]);

//     for (int k = 0; k < K; ++k) {
//         // Load B row of 4 elements: B[k, j .. j+3]
//         __m256d bvec = _mm256_loadu_pd(&B_tile[k * ldb]);

//         // For each A scalar in column k (A[i + r, k]) broadcast and fmadd
//         double a0 = A_tile[0 * lda + k];
//         double a1 = A_tile[1 * lda + k];
//         double a2 = A_tile[2 * lda + k];
//         double a3 = A_tile[3 * lda + k];

//         __m256d a0b = _mm256_broadcast_sd(&a0); c0 = _mm256_fmadd_pd(a0b, bvec, c0);
//         __m256d a1b = _mm256_broadcast_sd(&a1); c1 = _mm256_fmadd_pd(a1b, bvec, c1);
//         __m256d a2b = _mm256_broadcast_sd(&a2); c2 = _mm256_fmadd_pd(a2b, bvec, c2);
//         __m256d a3b = _mm256_broadcast_sd(&a3); c3 = _mm256_fmadd_pd(a3b, bvec, c3);
//     }

//     _mm256_storeu_pd(&C_tile[0 * ldc], c0);
//     _mm256_storeu_pd(&C_tile[1 * ldc], c1);
//     _mm256_storeu_pd(&C_tile[2 * ldc], c2);
//     _mm256_storeu_pd(&C_tile[3 * ldc], c3);
// }

// // scalar fallback for MxN small tiles (M,N <= 4)
// static inline void scalar_small(const double* __restrict__ A_tile,
//                                 const double* __restrict__ B_tile,
//                                 double* __restrict__ C_tile,
//                                 int lda, int ldb, int ldc,
//                                 int K, int M, int N)
// {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             double acc = 0.0;
//             for (int k = 0; k < K; ++k) acc += A_tile[i * lda + k] * B_tile[k * ldb + j];
//             C_tile[i * ldc + j] += acc;
//         }
//     }
// }

// // high-level blocked GEMM (row-major matrices)
// // A: N x N, B: N x N, C: N x N, lda/ldb/ldc are strides (>= N)
// void gemm_tiled_4x4(const double* A, const double* B, double* C,
//                     int N, int MC = 256, int KC = 128, int NC = 256,
//                     int lda = 0, int ldb = 0, int ldc = 0)
// {
//     if (!lda) lda = N;
//     if (!ldb) ldb = N;
//     if (!ldc) ldc = N;

//     #pragma omp parallel for collapse(2) schedule(dynamic)
//     for (int jc = 0; jc < N; jc += NC) {
//         for (int ic = 0; ic < N; ic += MC) {
//             for (int pc = 0; pc < N; pc += KC) {
//                 int jMax = min(jc + NC, N);
//                 int iMax = min(ic + MC, N);
//                 int pMax = min(pc + KC, N);
//                 int klen = pMax - pc;

//                 for (int j = jc; j < jMax; j += NR) {
//                     int j_rem = min(NR, jMax - j);
//                     for (int i = ic; i < iMax; i += MR) {
//                         int i_rem = min(MR, iMax - i);

//                         // pointers to the top-left of the current MRxNR block
//                         double* Cblk = &C[i * ldc + j];
//                         const double* Ablk = &A[i * lda + pc];
//                         const double* Bblk = &B[pc * ldb + j];

//                         if (i_rem == MR && j_rem == NR) {
//                             // full 4x4 micro-kernel
//                             microkernel4x4(Ablk, Bblk, Cblk, lda, ldb, ldc, klen);
//                         } else {
//                             // small tail: scalar fallback
//                             scalar_small(Ablk, Bblk, Cblk, lda, ldb, ldc, klen, i_rem, j_rem);
//                         }
//                     } // i
//                 } // j
//             } // pc
//         } // ic
//     } // jc
// }

// int main(int argc, char** argv)
// {
//     if (argc < 3) {
//         cerr << "Usage: " << argv[0] << " N num_threads [MC KC NC]\n";
//         return 1;
//     }
//     int N = atoi(argv[1]);
//     int T = atoi(argv[2]);
//     int MC = (argc > 3) ? atoi(argv[3]) : 256;
//     int KC = (argc > 4) ? atoi(argv[4]) : 128;
//     int NC = (argc > 5) ? atoi(argv[5]) : 256;

//     omp_set_dynamic(0);
//     omp_set_num_threads(T);

//     int pad = 0;
//     int lda = N + pad, ldb = N + pad, ldc = N + pad;

//     size_t ne = (size_t)N * (size_t)N;
//     double* A = (double*) aligned_malloc(sizeof(double) * ne);
//     double* B = (double*) aligned_malloc(sizeof(double) * ne);
//     double* C = (double*) aligned_malloc(sizeof(double) * ne);
//     if (!A || !B || !C) { cerr << "Allocation failed\n"; aligned_free(A); aligned_free(B); aligned_free(C); return 1; }

//     // init (simple deterministic pattern)
//     // for (int i = 0; i < N; ++i)
//     //     for (int j = 0; j < N; ++j) {
//     //         A[i * lda + j] = (double)((i + j) & 0xFF) / 17.0;
//     //         B[i * ldb + j] = (double)((i - j) & 0xFF) / 31.0;
//     //         C[i * ldc + j] = 0.0;
//     //     }

//     // // page-touch to reduce page faults during timed region
//     // #pragma omp parallel
//     // {
//     //     int tid = omp_get_thread_num();
//     //     int nth = omp_get_num_threads();
//     //     size_t chunk = (ne + nth - 1) / nth;
//     //     size_t start = (size_t)tid * chunk;
//     //     size_t end = min(ne, start + chunk);
//     //     for (size_t idx = start; idx < end; idx += 4096 / sizeof(double)) {
//     //         A[idx] = A[idx];
//     //         B[idx] = B[idx];
//     //         C[idx] = C[idx];
//     //     }
//     // }

    
//     // reproducible RNG
//     std::mt19937_64 rng(12345);
//     std::normal_distribution<double> dist(0.0, 1.0);
//     for (int i=0;i<N*N;i++) { A[i] = dist(rng); B[i] = dist(rng); C[i]=0.0; }

//     cout << fixed << setprecision(6);
//     cout << "N="<<N<<" threads="<<T<<" MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";

//     double t0 = omp_get_wtime();
//     gemm_tiled_4x4(A, B, C, N, MC, KC, NC, lda, ldb, ldc);
//     double t1 = omp_get_wtime();

//     double elapsed = t1 - t0;
//     long double checksum = 0.0L;
//     #pragma omp parallel for reduction(+ : checksum) schedule(static)
//     for (int i = 0; i < N; ++i)
//         for (int j = 0; j < N; ++j)
//             checksum += C[i * ldc + j];

//     double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);
//     cout << "Elapsed(s): " << elapsed << "  GFLOPs: " << gflops << "  checksum: " << setprecision(17) << (double)checksum << "\n";

//     aligned_free(A); aligned_free(B); aligned_free(C);
//     return 0;
// }

// // =1000 threads=8 MC=256 KC=128 NC=256
// // Elapsed(s): 0.018978  GFLOPs: 105.387887  checksum: 4950.57577906217284180

// //  Performance counter stats for './optimized/gemm_opt 1000 8':

// //             233.88 msec task-clock                       #    2.794 CPUs utilized             
// //                 15      context-switches                 #   64.136 /sec                      
// //                  8      cpu-migrations                   #   34.206 /sec                      
// //              6,025      page-faults                      #   25.761 K/sec                     
// //        704,946,805      cycles                           #    3.014 GHz                       
// //         28,652,990      stalled-cycles-frontend          #    4.06% frontend cycles idle      
// //      1,259,121,294      instructions                     #    1.79  insn per cycle            
// //                                                   #    0.02  stalled cycles per insn   
// //         74,467,709      branches                         #  318.405 M/sec                     
// //          2,719,292      branch-misses                    #    3.65% of all branches           

// //        0.083709402 seconds time elapsed

// //        0.214708000 seconds user
// //        0.020160000 seconds sys


// gemm_4x4_simple.cpp
// Simple, readable tiled GEMM with a 4x4 AVX2 micro-kernel and scalar tail fallback.
// - No packing, no autotune, minimal dependencies
// Compile:
//   g++ -O3 -march=native -mavx2 -mfma -fopenmp gemm_4x4_simple.cpp -o gemm_4x4_simple
//
// Run:
//   ./gemm_4x4_simple 1000 8 256 128 256
//
// Arguments:
//   N threads [MC KC NC]
// Defaults: MC=256, KC=128, NC=256

// #include <bits/stdc++.h>
// #include <immintrin.h>
// #include <omp.h>
// using namespace std;

// constexpr int MR = 4;
// constexpr int NR = 4;

// // aligned allocation
// static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
//     void* p = nullptr;
//     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
//     return p;
// }
// static inline void aligned_free(void* p) { free(p); }

// // ----------------------- micro-kernel (4x4, AVX2, permute-based) -----------------------
// // A_tile: pointer to block A[i..i+3, pc..pc+K-1] with stride lda (row-major per Ablk)
// // B_tile: pointer to block B[pc..pc+K-1, j..j+3] with stride ldb (row-major per Bblk)
// // C_tile: pointer to block C[i..i+3, j..j+3] with stride ldc
// // K: reduction length
// static inline void microkernel4x4(const double* __restrict__ A_tile,
//                                   const double* __restrict__ B_tile,
//                                   double* __restrict__ C_tile,
//                                   int lda, int ldb, int ldc,
//                                   int K)
// {
//     // Permute immediates for broadcasting lanes
//     const int P0 = 0x00, P1 = 0x55, P2 = 0xAA, P3 = 0xFF;

//     // Load C rows (each row has 4 contiguous columns)
//     __m256d c0 = _mm256_loadu_pd(&C_tile[0 * ldc]);
//     __m256d c1 = _mm256_loadu_pd(&C_tile[1 * ldc]);
//     __m256d c2 = _mm256_loadu_pd(&C_tile[2 * ldc]);
//     __m256d c3 = _mm256_loadu_pd(&C_tile[3 * ldc]);

//     for (int k = 0; k < K; ++k) {
//         // Load one row of B (4 doubles)
//         __m256d b = _mm256_loadu_pd(&B_tile[k * ldb]);

//         // Load 4 A scalars for this k as a vector: [a0 a1 a2 a3]
//         // Note: A_tile layout: row-major with stride lda; element A_tile[r*lda + k]
//         __m256d avec = _mm256_setr_pd(
//             A_tile[0 * lda + k],
//             A_tile[1 * lda + k],
//             A_tile[2 * lda + k],
//             A_tile[3 * lda + k]
//         );

//         // Create per-row vectors by permuting lanes and FMADD
//         __m256d a0v = _mm256_permute4x64_pd(avec, P0);
//         __m256d a1v = _mm256_permute4x64_pd(avec, P1);
//         __m256d a2v = _mm256_permute4x64_pd(avec, P2);
//         __m256d a3v = _mm256_permute4x64_pd(avec, P3);

//         c0 = _mm256_fmadd_pd(a0v, b, c0);
//         c1 = _mm256_fmadd_pd(a1v, b, c1);
//         c2 = _mm256_fmadd_pd(a2v, b, c2);
//         c3 = _mm256_fmadd_pd(a3v, b, c3);
//     }

//     _mm256_storeu_pd(&C_tile[0 * ldc], c0);
//     _mm256_storeu_pd(&C_tile[1 * ldc], c1);
//     _mm256_storeu_pd(&C_tile[2 * ldc], c2);
//     _mm256_storeu_pd(&C_tile[3 * ldc], c3);
// }

// // scalar fallback for small tiles (M,N <= 4)
// static inline void scalar_small(const double* __restrict__ Ablk,
//                                 const double* __restrict__ Bblk,
//                                 double* __restrict__ Cblk,
//                                 int lda, int ldb, int ldc,
//                                 int K, int M, int N)
// {
//     for (int i = 0; i < M; ++i)
//         for (int j = 0; j < N; ++j) {
//             double acc = 0.0;
//             for (int k = 0; k < K; ++k) acc += Ablk[i * lda + k] * Bblk[k * ldb + j];
//             Cblk[i * ldc + j] += acc;
//         }
// }

// // ----------------------- tiled GEMM (simple, readable) -----------------------
// void gemm_tiled_simple(const double* A, const double* B, double* C,
//                        int N, int MC = 256, int KC = 128, int NC = 256,
//                        int lda = 0, int ldb = 0, int ldc = 0)
// {
//     if (!lda) lda = N;
//     if (!ldb) ldb = N;
//     if (!ldc) ldc = N;

//     #pragma omp parallel for collapse(2) schedule(dynamic)
//     for (int jc = 0; jc < N; jc += NC) {
//         for (int ic = 0; ic < N; ic += MC) {
//             for (int pc = 0; pc < N; pc += KC) {
//                 int jMax = min(jc + NC, N);
//                 int iMax = min(ic + MC, N);
//                 int pMax = min(pc + KC, N);
//                 int klen = pMax - pc;

//                 for (int j = jc; j < jMax; j += NR) {
//                     int j_rem = min(NR, jMax - j);
//                     for (int i = ic; i < iMax; i += MR) {
//                         int i_rem = min(MR, iMax - i);

//                         const double* Ablk = &A[i * lda + pc];
//                         const double* Bblk = &B[pc * ldb + j];
//                         double* Cblk = &C[i * ldc + j];

//                         if (i_rem == MR && j_rem == NR) {
//                             microkernel4x4(Ablk, Bblk, Cblk, lda, ldb, ldc, klen);
//                         } else {
//                             scalar_small(Ablk, Bblk, Cblk, lda, ldb, ldc, klen, i_rem, j_rem);
//                         }
//                     } // i
//                 } // j
//             } // pc
//         } // ic
//     } // jc
// }

// // ----------------------- main (simple) -----------------------
// int main(int argc, char** argv)
// {
//     if (argc < 3) {
//         cerr << "Usage: " << argv[0] << " N num_threads [MC KC NC]\n";
//         return 1;
//     }
//     int N = atoi(argv[1]);
//     int T = atoi(argv[2]);
//     int MC = (argc > 3) ? atoi(argv[3]) : 256;
//     int KC = (argc > 4) ? atoi(argv[4]) : 128;
//     int NC = (argc > 5) ? atoi(argv[5]) : 256;

//     omp_set_dynamic(0);
//     omp_set_num_threads(T);

//     int lda = N, ldb = N, ldc = N;
//     size_t ne = (size_t)N * (size_t)N;

//     double* A = (double*) aligned_malloc(sizeof(double) * ne);
//     double* B = (double*) aligned_malloc(sizeof(double) * ne);
//     double* C = (double*) aligned_malloc(sizeof(double) * ne);
//     if (!A || !B || !C) { cerr << "Allocation failed\n"; return 1; }

//     // reproducible RNG init (compact form)
//     std::mt19937_64 rng(12345);
//     std::normal_distribution<double> dist(0.0, 1.0);
//     for (size_t i = 0; i < ne; ++i) { A[i] = dist(rng); B[i] = dist(rng); C[i] = 0.0; }

//     // simple page-touch warmup (parallel)
//     #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         int nth = omp_get_num_threads();
//         size_t chunk = (ne + nth - 1) / nth;
//         size_t start = (size_t)tid * chunk;
//         size_t end = min(ne, start + chunk);
//         for (size_t idx = start; idx < end; idx += 4096 / sizeof(double)) {
//             volatile double tmp = A[idx]; (void)tmp;
//             tmp = B[idx]; (void)tmp;
//             tmp = C[idx]; (void)tmp;
//         }
//     }

//     cout << "N="<<N<<" threads="<<T<<" MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";

//     double t0 = omp_get_wtime();
//     gemm_tiled_simple(A, B, C, N, MC, KC, NC, lda, ldb, ldc);
//     double t1 = omp_get_wtime();

//     double elapsed = t1 - t0;
//     long double checksum = 0.0L;
//     #pragma omp parallel for reduction(+ : checksum) schedule(static)
//     for (int i = 0; i < N; ++i)
//         for (int j = 0; j < N; ++j)
//             checksum += C[i * ldc + j];

//     double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);
//     cout << fixed << setprecision(6);
//     cout << "Elapsed(s): " << elapsed << "  GFLOPs: " << gflops << "  checksum: " << setprecision(17) << (double)checksum << "\n";

//     aligned_free(A); aligned_free(B); aligned_free(C);
//     return 0;
// }
