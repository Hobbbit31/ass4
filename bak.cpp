
// // #include <bits/stdc++.h>
// // #include <immintrin.h>
// // #include <omp.h>
// // using namespace std;

// // constexpr int MR = 4;
// // constexpr int NR = 4;

// // // aligned alloc
// // static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
// //     void* p = nullptr;
// //     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
// //     return p;
// // }
// // static inline void aligned_free(void* p) { free(p); }

// // // --- New Packing Functions ---

// // // Pack A panel (ilen x klen) into A_pack (col-panel format)
// // // A_pack layout: [A(i,k), A(i+1,k), A(i+2,k), A(i+3,k), A(i,k+1), A(i+1,k+1), ...]
// // static void pack_A_panel(const double* __restrict__ A,
// //                          double* __restrict__ A_pack,
// //                          int ic, int pc, int ilen, int klen, int lda)
// // {
// //     int pack_offset = 0;
// //     for (int i_p = 0; i_p < ilen; i_p += MR) {
// //         int i_rem = min(MR, ilen - i_p);
// //         for (int k_p = 0; k_p < klen; ++k_p) {
// //             for (int r = 0; r < i_rem; ++r) {
// //                 A_pack[pack_offset + r] = A[(ic + i_p + r) * lda + (pc + k_p)];
// //             }
// //             // Zero-pad the remainder of the panel
// //             for (int r = i_rem; r < MR; ++r) {
// //                 A_pack[pack_offset + r] = 0.0;
// //             }
// //             pack_offset += MR;
// //         }
// //     }
// // }

// // // Pack B panel (klen x jlen) into B_pack (row-major format)
// // static void pack_B_panel(const double* __restrict__ B,
// //                          double* __restrict__ B_pack,
// //                          int pc, int jc, int klen, int jlen, int ldb)
// // {
// //     int pack_offset = 0;
// //     for (int k_p = 0; k_p < klen; ++k_p) {
// //         // Use memcpy for fast row copies
// //         memcpy(&B_pack[pack_offset], &B[(pc + k_p) * ldb + jc], jlen * sizeof(double));
// //         pack_offset += jlen;
// //     }
// // }


// // // --- New 4x4 AVX2 micro-kernel (operates on packed buffers) ---
// // // A_tile: pointer to a packed MRxK (4xK) panel of A
// // // B_tile: pointer to the start of a packed KxNC (Kxjlen) panel of B
// // static inline void microkernel4x4_packed(const double* __restrict__ A_tile,
// //                                          const double* __restrict__ B_tile,
// //                                          double* __restrict__ C_tile,
// //                                          int ldb_pack, // packed B stride (jlen)
// //                                          int ldc,
// //                                          int K)
// // {
// //     // load C rows
// //     __m256d c0 = _mm256_loadu_pd(&C_tile[0 * ldc]);
// //     __m256d c1 = _mm256_loadu_pd(&C_tile[1 * ldc]);
// //     __m256d c2 = _mm256_loadu_pd(&C_tile[2 * ldc]);
// //     __m256d c3 = _mm256_loadu_pd(&C_tile[3 * ldc]);

// //     const double* A_ptr = A_tile;
// //     const double* B_ptr = B_tile;

// //     for (int k = 0; k < K; ++k) {
// //         // Load B row of 4 elements: B[k, j .. j+3]
// //         __m256d bvec = _mm256_loadu_pd(B_ptr);

// //         // Load A column-panel: A[i..i+3, k]
// //         // This is now ONE contiguous vector load
// //         __m256d avec = _mm256_loadu_pd(A_ptr); // [a0, a1, a2, a3]

// //         // Broadcast a0 and fmadd
// //         c0 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec, _MM_SHUFFLE(0,0,0,0)), bvec, c0);
// //         // Broadcast a1 and fmadd
// //         c1 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec, _MM_SHUFFLE(1,1,1,1)), bvec, c1);
// //         // Broadcast a2 and fmadd
// //         c2 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec, _MM_SHUFFLE(2,2,2,2)), bvec, c2);
// //         // Broadcast a3 and fmadd
// //         c3 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec, _MM_SHUFFLE(3,3,3,3)), bvec, c3);

// //         A_ptr += MR;        // Move to next k-column in A_pack
// //         B_ptr += ldb_pack;  // Move to next k-row in B_pack
// //     }

// //     _mm256_storeu_pd(&C_tile[0 * ldc], c0);
// //     _mm256_storeu_pd(&C_tile[1 * ldc], c1);
// //     _mm256_storeu_pd(&C_tile[2 * ldc], c2);
// //     _mm256_storeu_pd(&C_tile[3 * ldc], c3);
// // }

// // // --- New scalar fallback (operates on packed buffers) ---
// // static inline void scalar_small_packed(const double* __restrict__ A_tile,
// //                                        const double* __restrict__ B_tile,
// //                                        double* __restrict__ C_tile,
// //                                        int ldb_pack, int ldc,
// //                                        int K, int M, int N)
// // {
// //     // A_tile is col-paneled (K-outer, MR-inner)
// //     // B_tile is row-major (K-outer, NR-inner)
// //     for (int i = 0; i < M; ++i) {
// //         for (int j = 0; j < N; ++j) {
// //             double acc = 0.0;
// //             for (int k = 0; k < K; ++k) {
// //                 // A[i,k] * B[k,j]
// //                 acc += A_tile[k * MR + i] * B_tile[k * ldb_pack + j];
// //             }
// //             C_tile[i * ldc + j] += acc;
// //         }
// //     }
// // }


// // // // --- Updated high-level blocked GEMM ---
// // // void gemm_tiled_4x4(const double* A, const double* B, double* C,
// // //                     int N, int MC = 256, int KC = 128, int NC = 256,
// // //                     int lda = 0, int ldb = 0, int ldc = 0)
// // // {
// // //     if (!lda) lda = N;
// // //     if (!ldb) ldb = N;
// // //     if (!ldc) ldc = N;

// // //     // Allocate packing buffers per-thread
// // //     #pragma omp parallel
// // //     {
// // //         // These buffers are sized to the L2/L3 block sizes
// // //         double* A_pack = (double*) aligned_malloc(sizeof(double) * MC * KC);
// // //         double* B_pack = (double*) aligned_malloc(sizeof(double) * KC * NC);
// // //         if (!A_pack || !B_pack) { /* handle error */ }

// // //         #pragma omp for collapse(2) schedule(dynamic)
// // //         for (int jc = 0; jc < N; jc += NC) {
// // //             for (int ic = 0; ic < N; ic += MC) {
                
// // //                 int jMax = min(jc + NC, N); int jlen = jMax - jc;
// // //                 int iMax = min(ic + MC, N); int ilen = iMax - ic;

// // //                 for (int pc = 0; pc < N; pc += KC) {
// // //                     int pMax = min(pc + KC, N); int klen = pMax - pc;

// // //                     // Pack the B panel (klen x jlen)
// // //                     pack_B_panel(B, B_pack, pc, jc, klen, jlen, ldb);
                    
// // //                     // Pack the A panel (ilen x klen)
// // //                     pack_A_panel(A, A_pack, ic, pc, ilen, klen, lda);

// // //                     for (int j = 0; j < jlen; j += NR) {
// // //                         int j_rem = min(NR, jlen - j);
// // //                         for (int i = 0; i < ilen; i += MR) {
// // //                             int i_rem = min(MR, ilen - i);

// // //                             // Pointers to the top-left of the current MRxNR block
// // //                             double* Cblk = &C[(ic+i) * ldc + (jc+j)];
                            
// // //                             // A_pack is (i/MR)-outer, k-middle, (i%MR)-inner
// // //                             const double* Ablk = &A_pack[ (i / MR) * (klen * MR) ];
// // //                             // B_pack is k-outer, j-inner
// // //                             const double* Bblk = &B_pack[ j ]; // &B_pack[0, j]

// // //                             if (i_rem == MR && j_rem == NR) {
// // //                                 // full 4x4 micro-kernel
// // //                                 microkernel4x4_packed(Ablk, Bblk, Cblk, jlen, ldc, klen);
// // //                             } else {
// // //                                 // small tail: scalar fallback (on packed data)
// // //                                 scalar_small_packed(Ablk, Bblk, Cblk, jlen, ldc, klen, i_rem, j_rem);
// // //                             }
// // //                         } // i
// // //                     } // j
// // //                 } // pc
// // //             } // ic
// // //         } // jc

// // //         aligned_free(A_pack);
// // //         aligned_free(B_pack);
// // //     } // omp parallel
// // // }

// // void gemm_tiled_4x4(const double* A, const double* B, double* C,
// //                     int N, int MC = 256, int KC = 128, int NC = 256,
// //                     int lda = 0, int ldb = 0, int ldc = 0)
// // {
// //     if (!lda) lda = N;
// //     if (!ldb) ldb = N;
// //     if (!ldc) ldc = N;

// //     #pragma omp parallel
// //     {
// //         double* A_pack = (double*) aligned_malloc(sizeof(double) * MC * KC);
// //         double* B_pack = (double*) aligned_malloc(sizeof(double) * KC * NC);
// //         if (!A_pack || !B_pack) { /* handle error */ }

// //         // New Loop Order: jc, pc, ic
// //         #pragma omp for schedule(dynamic)
// //         for (int jc = 0; jc < N; jc += NC) {
// //             int jMax = min(jc + NC, N); int jlen = jMax - jc;

// //             for (int pc = 0; pc < N; pc += KC) {
// //                 int pMax = min(pc + KC, N); int klen = pMax - pc;

// //                 // Pack B panel (klen x jlen) ONCE for this jc, pc block
// //                 // This will be reused for all 'ic' blocks
// //                 pack_B_panel(B, B_pack, pc, jc, klen, jlen, ldb);

// //                 for (int ic = 0; ic < N; ic += MC) {
// //                     int iMax = min(ic + MC, N); int ilen = iMax - ic;
                    
// //                     // Pack A panel (ilen x klen) for this 'ic' block
// //                     pack_A_panel(A, A_pack, ic, pc, ilen, klen, lda);

// //                     // --- Inner loops compute over packed buffers ---
// //                     for (int j = 0; j < jlen; j += NR) {
// //                         int j_rem = min(NR, jlen - j);
// //                         for (int i = 0; i < ilen; i += MR) {
// //                             int i_rem = min(MR, ilen - i);

// //                             double* Cblk = &C[(ic+i) * ldc + (jc+j)];
// //                             const double* Ablk = &A_pack[ (i / MR) * (klen * MR) ];
// //                             const double* Bblk = &B_pack[ j ];

// //                             if (i_rem == MR && j_rem == NR) {
// //                                 microkernel4x4_packed(Ablk, Bblk, Cblk, jlen, ldc, klen);
// //                             } else {
// //                                 scalar_small_packed(Ablk, Bblk, Cblk, jlen, ldc, klen, i_rem, j_rem);
// //                             }
// //                         } // i
// //                     } // j
// //                 } // ic
// //             } // pc
// //         } // jc

// //         aligned_free(A_pack);
// //         aligned_free(B_pack);
// //     } // omp parallel
// // }

// // int main(int argc, char** argv)
// // {
// //     if (argc < 3) {
// //         cerr << "Usage: " << argv[0] << " N num_threads [MC KC NC]\n";
// //         return 1;
// //     }
// //     int N = atoi(argv[1]);
// //     int T = atoi(argv[2]);
// //     int MC = (argc > 3) ? atoi(argv[3]) : 256;
// //     int KC = (argc > 4) ? atoi(argv[4]) : 128;
// //     int NC = (argc > 5) ? atoi(argv[5]) : 256;

// //     omp_set_dynamic(0);
// //     omp_set_num_threads(T);

// //     // Using lda=N for simplicity with packing routines
// //     int lda = N, ldb = N, ldc = N;

// //     size_t ne = (size_t)N * (size_t)N;
// //     double* A = (double*) aligned_malloc(sizeof(double) * ne);
// //     double* B = (double*) aligned_malloc(sizeof(double) * ne);
// //     double* C = (double*) aligned_malloc(sizeof(double) * ne);
// //     if (!A || !B || !C) { cerr << "Allocation failed\n"; aligned_free(A); aligned_free(B); aligned_free(C); return 1; }

// //     // reproducible RNG
// //     std::mt19937_64 rng(12345);
// //     std::normal_distribution<double> dist(0.0, 1.0);
// //     for (int i=0;i<N*N;i++) { A[i] = dist(rng); B[i] = dist(rng); C[i]=0.0; }

// //     // page-touch to reduce page faults during timed region
// //     #pragma omp parallel
// //     {
// //         int tid = omp_get_thread_num();
// //         int nth = omp_get_num_threads();
// //         size_t chunk = (ne + nth - 1) / nth;
// //         size_t start = (size_t)tid * chunk;
// //         size_t end = min(ne, start + chunk);
// //         for (size_t idx = start; idx < end; idx += 4096 / sizeof(double)) {
// //             // Volatile to prevent compiler from optimizing away
// //             ((volatile double*)A)[idx] = ((volatile double*)A)[idx];
// //             ((volatile double*)B)[idx] = ((volatile double*)B)[idx];
// //             ((volatile double*)C)[idx] = ((volatile double*)C)[idx];
// //         }
// //     }

    
// //     cout << fixed << setprecision(6);
// //     cout << "N="<<N<<" threads="<<T<<" MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";

// //     double t0 = omp_get_wtime();
// //     gemm_tiled_4x4(A, B, C, N, MC, KC, NC, lda, ldb, ldc);
// //     double t1 = omp_get_wtime();

// //     double elapsed = t1 - t0;
// //     long double checksum = 0.0L;
// //     #pragma omp parallel for reduction(+ : checksum) schedule(static)
// //     for (int i = 0; i < N; ++i)
// //         for (int j = 0; j < N; ++j)
// //             checksum += C[i * ldc + j];

// //     double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);
// //     cout << "Elapsed(s): " << elapsed << "  GFLOPs: " << gflops << "  checksum: " << setprecision(17) << (double)checksum << "\n";

// //     aligned_free(A); aligned_free(B); aligned_free(C);
// //     return 0;
// // }

// #include <bits/stdc++.h>
// #include <immintrin.h>
// #include <omp.h>
// using namespace std;

// // --- MR is now 8 ---
// constexpr int MR = 8;
// constexpr int NR = 4;

// // aligned alloc
// static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
//     void* p = nullptr;
//     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
//     return p;
// }
// static inline void aligned_free(void* p) { free(p); }

// // --- Packing Functions (Unchanged, they adapt to MR) ---
// static void pack_A_panel(const double* __restrict__ A,
//                          double* __restrict__ A_pack,
//                          int ic, int pc, int ilen, int klen, int lda)
// {
//     int pack_offset = 0;
//     for (int i_p = 0; i_p < ilen; i_p += MR) {
//         int i_rem = min(MR, ilen - i_p);
//         for (int k_p = 0; k_p < klen; ++k_p) {
//             for (int r = 0; r < i_rem; ++r) {
//                 A_pack[pack_offset + r] = A[(ic + i_p + r) * lda + (pc + k_p)];
//             }
//             for (int r = i_rem; r < MR; ++r) {
//                 A_pack[pack_offset + r] = 0.0;
//             }
//             pack_offset += MR;
//         }
//     }
// }
// static void pack_B_panel(const double* __restrict__ B,
//                          double* __restrict__ B_pack,
//                          int pc, int jc, int klen, int jlen, int ldb)
// {
//     int pack_offset = 0;
//     for (int k_p = 0; k_p < klen; ++k_p) {
//         memcpy(&B_pack[pack_offset], &B[(pc + k_p) * ldb + jc], jlen * sizeof(double));
//         pack_offset += jlen;
//     }
// }


// // --- New 8x4 AVX2 micro-kernel ---
// static inline void microkernel8x4_packed(const double* __restrict__ A_tile,
//                                          const double* __restrict__ B_tile,
//                                          double* __restrict__ C_tile,
//                                          int ldb_pack, // packed B stride (jlen)
//                                          int ldc,
//                                          int K)
// {
//     // Load C rows
//     __m256d c0 = _mm256_loadu_pd(&C_tile[0 * ldc]);
//     __m256d c1 = _mm256_loadu_pd(&C_tile[1 * ldc]);
//     __m256d c2 = _mm256_loadu_pd(&C_tile[2 * ldc]);
//     __m256d c3 = _mm256_loadu_pd(&C_tile[3 * ldc]);
//     __m256d c4 = _mm256_loadu_pd(&C_tile[4 * ldc]);
//     __m256d c5 = _mm256_loadu_pd(&C_tile[5 * ldc]);
//     __m256d c6 = _mm256_loadu_pd(&C_tile[6 * ldc]);
//     __m256d c7 = _mm256_loadu_pd(&C_tile[7 * ldc]);

//     const double* A_ptr = A_tile;
//     const double* B_ptr = B_tile;

//     for (int k = 0; k < K; ++k) {
//         // Load B row of 4 elements: B[k, j .. j+3]
//         __m256d bvec = _mm256_loadu_pd(B_ptr);

//         // Load A column-panel: A[i..i+7, k]
//         __m256d avec0 = _mm256_loadu_pd(A_ptr);      // [a0, a1, a2, a3]
//         __m256d avec1 = _mm256_loadu_pd(A_ptr + 4);  // [a4, a5, a6, a7]

//         // Broadcast and fmadd for a0-a3
//         c0 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec0, _MM_SHUFFLE(0,0,0,0)), bvec, c0);
//         c1 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec0, _MM_SHUFFLE(1,1,1,1)), bvec, c1);
//         c2 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec0, _MM_SHUFFLE(2,2,2,2)), bvec, c2);
//         c3 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec0, _MM_SHUFFLE(3,3,3,3)), bvec, c3);
        
//         // Broadcast and fmadd for a4-a7
//         c4 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec1, _MM_SHUFFLE(0,0,0,0)), bvec, c4);
//         c5 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec1, _MM_SHUFFLE(1,1,1,1)), bvec, c5);
//         c6 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec1, _MM_SHUFFLE(2,2,2,2)), bvec, c6);
//         c7 = _mm256_fmadd_pd(_mm256_permute4x64_pd(avec1, _MM_SHUFFLE(3,3,3,3)), bvec, c7);

//         A_ptr += MR;        // Move to next k-column in A_pack
//         B_ptr += ldb_pack;  // Move to next k-row in B_pack
//     }

//     // Store C rows
//     _mm256_storeu_pd(&C_tile[0 * ldc], c0);
//     _mm256_storeu_pd(&C_tile[1 * ldc], c1);
//     _mm256_storeu_pd(&C_tile[2 * ldc], c2);
//     _mm256_storeu_pd(&C_tile[3 * ldc], c3);
//     _mm256_storeu_pd(&C_tile[4 * ldc], c4);
//     _mm256_storeu_pd(&C_tile[5 * ldc], c5);
//     _mm256_storeu_pd(&C_tile[6 * ldc], c6);
//     _mm256_storeu_pd(&C_tile[7 * ldc], c7);
// }

// // --- Scalar Fallback (Unchanged, adapts to MR) ---
// static inline void scalar_small_packed(const double* __restrict__ A_tile,
//                                        const double* __restrict__ B_tile,
//                                        double* __restrict__ C_tile,
//                                        int ldb_pack, int ldc,
//                                        int K, int M, int N)
// {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             double acc = 0.0;
//             for (int k = 0; k < K; ++k) {
//                 acc += A_tile[k * MR + i] * B_tile[k * ldb_pack + j];
//             }
//             C_tile[i * ldc + j] += acc;
//         }
//     }
// }


// // --- Updated high-level blocked GEMM (jc, pc, ic loop order) ---
// void gemm_tiled_8x4(const double* A, const double* B, double* C,
//                     int N, int MC = 256, int KC = 128, int NC = 256,
//                     int lda = 0, int ldb = 0, int ldc = 0)
// {
//     if (!lda) lda = N;
//     if (!ldb) ldb = N;
//     if (!ldc) ldc = N;

//     #pragma omp parallel
//     {
//         double* A_pack = (double*) aligned_malloc(sizeof(double) * MC * KC);
//         double* B_pack = (double*) aligned_malloc(sizeof(double) * KC * NC);
//         if (!A_pack || !B_pack) { /* handle error */ }

//         #pragma omp for schedule(dynamic)
//         for (int jc = 0; jc < N; jc += NC) {
//             int jMax = min(jc + NC, N); int jlen = jMax - jc;

//             for (int pc = 0; pc < N; pc += KC) {
//                 int pMax = min(pc + KC, N); int klen = pMax - pc;
//                 pack_B_panel(B, B_pack, pc, jc, klen, jlen, ldb);

//                 for (int ic = 0; ic < N; ic += MC) {
//                     int iMax = min(ic + MC, N); int ilen = iMax - ic;
//                     pack_A_panel(A, A_pack, ic, pc, ilen, klen, lda);

//                     for (int j = 0; j < jlen; j += NR) {
//                         int j_rem = min(NR, jlen - j);
//                         for (int i = 0; i < ilen; i += MR) {
//                             int i_rem = min(MR, ilen - i);

//                             double* Cblk = &C[(ic+i) * ldc + (jc+j)];
//                             const double* Ablk = &A_pack[ (i / MR) * (klen * MR) ];
//                             const double* Bblk = &B_pack[ j ];

//                             // --- Kernel call updated ---
//                             // Now checks for 8x4
//                             if (i_rem == MR && j_rem == NR) {
//                                 microkernel8x4_packed(Ablk, Bblk, Cblk, jlen, ldc, klen);
//                             } else {
//                                 scalar_small_packed(Ablk, Bblk, Cblk, jlen, ldc, klen, i_rem, j_rem);
//                             }
//                         } // i
//                     } // j
//                 } // ic
//             } // pc
//         } // jc

//         aligned_free(A_pack);
//         aligned_free(B_pack);
//     } // omp parallel
// }

// // --- Main Function (Unchanged except call to gemm_tiled_8x4) ---
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
//     if (!A || !B || !C) { cerr << "Allocation failed\n"; aligned_free(A); aligned_free(B); aligned_free(C); return 1; }

//     std::mt19937_64 rng(12345);
//     std::normal_distribution<double> dist(0.0, 1.0);
//     for (int i=0;i<N*N;i++) { A[i] = dist(rng); B[i] = dist(rng); C[i]=0.0; }

//     // Page-touch
//     #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         int nth = omp_get_num_threads();
//         size_t chunk = (ne + nth - 1) / nth;
//         size_t start = (size_t)tid * chunk;
//         size_t end = min(ne, start + chunk);
//         for (size_t idx = start; idx < end; idx += 4096 / sizeof(double)) {
//             ((volatile double*)A)[idx] = ((volatile double*)A)[idx];
//             ((volatile double*)B)[idx] = ((volatile double*)B)[idx];
//             ((volatile double*)C)[idx] = ((volatile double*)C)[idx];
//         }
//     }

    
//     cout << fixed << setprecision(6);
//     cout << "Running with: N="<<N<<" threads="<<T<<" MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";

//     double t0 = omp_get_wtime();
//     // --- Call updated function ---
//     gemm_tiled_8x4(A, B, C, N, MC, KC, NC, lda, ldb, ldc);
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