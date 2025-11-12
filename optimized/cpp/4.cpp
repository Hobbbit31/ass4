// // #include <bits/stdc++.h>
// // #include <immintrin.h>
// // #include <omp.h>
// // using namespace std;

// // // ======================================================
// // // =============== SIMD FEATURE CHECK ===================
// // // ======================================================
// // void check_simd_support() {
// //     unsigned a,b,c,d;
// //     bool sse=0,avx=0,avx2=0,avx512=0;

// //     __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(1));
// //     sse = (d >> 25) & 1;
// //     if ((c & (1<<27)) && (c & (1<<28))) {
// //         unsigned x0a,x0d;
// //         __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
// //         if ((x0a & 6) == 6) avx = 1;
// //     }

// //     __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(7));
// //     if (avx && (b & (1<<5))) avx2 = 1;
// //     if ((b & (1<<16))) {
// //         unsigned x0a,x0d;
// //         __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
// //         if ((x0a & 0xE0) == 0xE0) avx512 = 1;
// //     }

// //     cout << "Detected SIMD support → ";
// //     cout << "SSE=" << sse << " AVX=" << avx << " AVX2=" << avx2 << " AVX512=" << avx512 << "\n";
// //     if (avx512) cout << "→ Best available SIMD: AVX-512\n";
// //     else if (avx2) cout << "→ Best available SIMD: AVX2\n";
// //     else if (avx) cout << "→ Best available SIMD: AVX\n";
// //     else cout << "→ Fallback: SSE or Scalar\n";
// // }

// // // ======================================================
// // // =============== GEMM IMPLEMENTATION ==================
// // // ======================================================
// // constexpr int MR = 4;
// // constexpr int NR = 4;

// // static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
// //     void* p = nullptr;
// //     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
// //     return p;
// // }
// // static inline void aligned_free(void* p) { free(p); }

// // static inline void microkernel4x4(const double* __restrict__ A_tile,const double* __restrict__ B_tile,double* __restrict__ C_tile,int lda, int ldb, int ldc,int K)
// // {
// //     __m256d c0 = _mm256_loadu_pd(&C_tile[0 * ldc]);
// //     __m256d c1 = _mm256_loadu_pd(&C_tile[1 * ldc]);
// //     __m256d c2 = _mm256_loadu_pd(&C_tile[2 * ldc]);
// //     __m256d c3 = _mm256_loadu_pd(&C_tile[3 * ldc]);

// //     for (int k = 0; k < K; ++k) {
// //         __m256d bvec = _mm256_loadu_pd(&B_tile[k * ldb]);
// //         double a0 = A_tile[0 * lda + k];
// //         double a1 = A_tile[1 * lda + k];
// //         double a2 = A_tile[2 * lda + k];
// //         double a3 = A_tile[3 * lda + k];

// //         c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a0), bvec, c0);
// //         c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a1), bvec, c1);
// //         c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a2), bvec, c2);
// //         c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a3), bvec, c3);
// //     }

// //     _mm256_storeu_pd(&C_tile[0 * ldc], c0);
// //     _mm256_storeu_pd(&C_tile[1 * ldc], c1);
// //     _mm256_storeu_pd(&C_tile[2 * ldc], c2);
// //     _mm256_storeu_pd(&C_tile[3 * ldc], c3);
// // }

// // static inline void scalar_small(const double* __restrict__ A_tile, const double* __restrict__ B_tile,double* __restrict__ C_tile,int lda, int ldb, int ldc, int K, int M, int N)
// // {
// //     for (int i = 0; i < M; ++i)
// //         for (int j = 0; j < N; ++j) {
// //             double acc = 0.0;
// //             for (int k = 0; k < K; ++k)
// //                 acc += A_tile[i * lda + k] * B_tile[k * ldb + j];
// //             C_tile[i * ldc + j] += acc;
// //         }
// // }

// // void gemm_tiled_4x4(const double* A, const double* B, double* C,int N, int MC = 256, int KC = 128, int NC = 256,int lda = 0, int ldb = 0, int ldc = 0)
// // {
// //     if (!lda) lda = N;
// //     if (!ldb) ldb = N;
// //     if (!ldc) ldc = N;

// //     // dynamic
// //     #pragma omp parallel for collapse(2) schedule(static)
// //     for (int jc = 0; jc < N; jc += NC) {
// //         for (int ic = 0; ic < N; ic += MC) {
// //             for (int pc = 0; pc < N; pc += KC) {
// //                 int jMax = min(jc + NC, N);
// //                 int iMax = min(ic + MC, N);
// //                 int pMax = min(pc + KC, N);
// //                 int klen = pMax - pc;

// //                 for (int j = jc; j < jMax; j += NR) {
// //                     int j_rem = min(NR, jMax - j);
// //                     for (int i = ic; i < iMax; i += MR) {
// //                         int i_rem = min(MR, iMax - i);

// //                         double* Cblk = &C[i * ldc + j];
// //                         const double* Ablk = &A[i * lda + pc];
// //                         const double* Bblk = &B[pc * ldb + j];

// //                         if (i_rem == MR && j_rem == NR)
// //                             microkernel4x4(Ablk, Bblk, Cblk, lda, ldb, ldc, klen);
// //                         else
// //                             scalar_small(Ablk, Bblk, Cblk, lda, ldb, ldc, klen, i_rem, j_rem);
// //                     }
// //                 }
// //             }
// //         }
// //     }
// // }

// // // ======================================================
// // // ==================== MAIN ============================
// // // ======================================================
// // int main(int argc, char** argv)
// // {
// //     check_simd_support(); // ← Detect SIMD capabilities

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

// //     size_t ne = (size_t)N * (size_t)N;
// //     double *A = (double*)aligned_malloc(sizeof(double) * ne);
// //     double *B = (double*)aligned_malloc(sizeof(double) * ne);
// //     double *C = (double*)aligned_malloc(sizeof(double) * ne);

// //     std::mt19937_64 rng(12345);
// //     std::normal_distribution<double> dist(0.0, 1.0);
// //     for (int i=0; i<N*N; i++) { A[i]=dist(rng); B[i]=dist(rng); C[i]=0.0; }

// //     cout << fixed << setprecision(6);
// //     cout << "N="<<N<<" threads="<<T<<" MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";

// //     double t0 = omp_get_wtime();
// //     gemm_tiled_4x4(A, B, C, N, MC, KC, NC);
// //     double t1 = omp_get_wtime();

// //     double elapsed = t1 - t0;
// //     long double checksum = 0.0L;
// //     #pragma omp parallel for reduction(+ : checksum)
// //     for (int i=0;i<N;i++)
// //         for (int j=0;j<N;j++)
// //             checksum += C[i*N + j];

// //     double gflops = (2.0 * N * N * N) / (elapsed * 1e9);
// //     cout << "Elapsed(s): " << elapsed << "  GFLOPs: " << gflops
// //          << "  checksum: " << setprecision(2) << (double)checksum << "\n";

// //     aligned_free(A); aligned_free(B); aligned_free(C);
// //     return 0;
// // }


// #include <bits/stdc++.h>
// #include <immintrin.h>
// #include <omp.h>
// using namespace std;

// // ======================================================
// // =============== SIMD FEATURE CHECK ===================
// // ======================================================
// void check_simd_support() {
//     unsigned a,b,c,d;
//     bool sse=0,avx=0,avx2=0,avx512=0;

//     __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(1));
//     sse = (d >> 25) & 1;
//     if ((c & (1<<27)) && (c & (1<<28))) {
//         unsigned x0a,x0d;
//         __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
//         if ((x0a & 6) == 6) avx = 1;
//     }

//     __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(7));
//     if (avx && (b & (1<<5))) avx2 = 1;
//     if ((b & (1<<16))) {
//         unsigned x0a,x0d;
//         __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
//         if ((x0a & 0xE0) == 0xE0) avx512 = 1;
//     }

//     cout << "Detected SIMD support → ";
//     cout << "SSE=" << sse << " AVX=" << avx << " AVX2=" << avx2 << " AVX512=" << avx512 << "\n";
//     if (avx512) cout << "→ Best available SIMD: AVX-512\n";
//     else if (avx2) cout << "→ Best available SIMD: AVX2\n";
//     else if (avx) cout << "→ Best available SIMD: AVX\n";
//     else cout << "→ Fallback: SSE or Scalar\n";
// }

// // ======================================================
// // =============== GEMM IMPLEMENTATION ==================
// // ======================================================
// constexpr int MR = 4;
// constexpr int NR = 4;

// static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
//     void* p = nullptr;
//     if (posix_memalign(&p, align, bytes) != 0) return nullptr;
//     return p;
// }
// static inline void aligned_free(void* p) { free(p); }

// static inline void microkernel4x4(const double* __restrict__ A_tile, const double* __restrict__ B_tile, double* __restrict__ C_tile,int lda, int ldb, int ldc, int K)
// {
//     __m256d c0 = _mm256_load_pd(&C_tile[0 * ldc]);
//     __m256d c1 = _mm256_load_pd(&C_tile[1 * ldc]);
//     __m256d c2 = _mm256_load_pd(&C_tile[2 * ldc]);
//     __m256d c3 = _mm256_load_pd(&C_tile[3 * ldc]);

//     for (int k = 0; k < K; ++k) {
//         __m256d bvec = _mm256_load_pd(&B_tile[k * ldb]);
//         double a0 = A_tile[0 * lda + k];
//         double a1 = A_tile[1 * lda + k];
//         double a2 = A_tile[2 * lda + k];
//         double a3 = A_tile[3 * lda + k];

//         c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a0), bvec, c0);
//         c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a1), bvec, c1);
//         c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a2), bvec, c2);
//         c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a3), bvec, c3);
//     }

//     _mm256_store_pd(&C_tile[0 * ldc], c0);
//     _mm256_store_pd(&C_tile[1 * ldc], c1);
//     _mm256_store_pd(&C_tile[2 * ldc], c2);
//     _mm256_store_pd(&C_tile[3 * ldc], c3);
// }

// static inline void scalar_small(const double* __restrict__ A_tile, const double* __restrict__ B_tile,double* __restrict__ C_tile,int lda, int ldb, int ldc, int K, int M, int N)
// {
//     for (int i = 0; i < M; ++i)
//         for (int j = 0; j < N; ++j) {
//             double acc = 0.0;
//             for (int k = 0; k < K; ++k)
//                 acc += A_tile[i * lda + k] * B_tile[k * ldb + j];
//             C_tile[i * ldc + j] += acc;
//         }
// }

// void gemm_tiled_4x4(const double* A, const double* B, double* C,int N, int MC = 256, int KC = 128, int NC = 256,int lda = 0, int ldb = 0, int ldc = 0)
// {
//     if (!lda) lda = N;
//     if (!ldb) ldb = N;
//     if (!ldc) ldc = N;

//     #pragma omp parallel for collapse(2) schedule(static)
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

//                         double* Cblk = &C[i * ldc + j];
//                         const double* Ablk = &A[i * lda + pc];
//                         const double* Bblk = &B[pc * ldb + j];

//                         if (i_rem == MR && j_rem == NR)
//                             microkernel4x4(Ablk, Bblk, Cblk, lda, ldb, ldc, klen);
//                         else
//                             scalar_small(Ablk, Bblk, Cblk, lda, ldb, ldc, klen, i_rem, j_rem);
//                     }
//                 }
//             }
//         }
//     }
// }

// // ======================================================
// // ==================== MAIN ============================
// // ======================================================
// int main(int argc, char** argv)
// {
//     check_simd_support(); // ← Detect SIMD capabilities

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

//     size_t ne = (size_t)N * (size_t)N;
//     double *A = (double*)aligned_malloc(sizeof(double) * ne);
//     double *B = (double*)aligned_malloc(sizeof(double) * ne);
//     double *C = (double*)aligned_malloc(sizeof(double) * ne);

//     std::mt19937_64 rng(12345);
//     std::normal_distribution<double> dist(0.0, 1.0);
//     for (int i=0; i<N*N; i++) { A[i]=dist(rng); B[i]=dist(rng); C[i]=0.0; }

//     cout << fixed << setprecision(6);
//     cout << "N="<<N<<" threads="<<T<<" MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";

//     double t0 = omp_get_wtime();
//     gemm_tiled_4x4(A, B, C, N, MC, KC, NC);
//     double t1 = omp_get_wtime();

//     double elapsed = t1 - t0;
//     long double checksum = 0.0L;
//     #pragma omp parallel for reduction(+ : checksum)
//     for (int i=0;i<N;i++)
//         for (int j=0;j<N;j++)
//             checksum += C[i*N + j];

//     double gflops = (2.0 * N * N * N) / (elapsed * 1e9);
//     cout << "Elapsed(s): " << elapsed << "  GFLOPs: " << gflops
//          << "  checksum: " << setprecision(2) << (double)checksum << "\n";

//     // aligned_free(A); aligned_free(B); aligned_free(C);
//     return 0;
// }

#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
using namespace std;

// ======================================================
// =============== SIMD FEATURE CHECK ===================
// ======================================================
void check_simd_support() {
    unsigned a,b,c,d;
    bool sse=0,avx=0,avx2=0,avx512=0;

    __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(1));
    sse = (d >> 25) & 1;
    if ((c & (1<<27)) && (c & (1<<28))) {
        unsigned x0a,x0d;
        __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
        if ((x0a & 6) == 6) avx = 1;
    }

    __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(7));
    if (avx && (b & (1<<5))) avx2 = 1;
    if ((b & (1<<16))) {
        unsigned x0a,x0d;
        __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
        if ((x0a & 0xE0) == 0xE0) avx512 = 1;
    }

    cout << "Detected SIMD support → ";
    cout << "SSE=" << sse << " AVX=" << avx << " AVX2=" << avx2 << " AVX512=" << avx512 << "\n";
    if (avx512) cout << "→ Best available SIMD: AVX-512\n";
    else if (avx2) cout << "→ Best available SIMD: AVX2\n";
    else if (avx) cout << "→ Best available SIMD: AVX\n";
    else cout << "→ Fallback: SSE or Scalar\n";
}

// ======================================================
// =============== GEMM IMPLEMENTATION ==================
// ======================================================
constexpr int MR = 4;
constexpr int NR = 4;

static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
}
static inline void aligned_free(void* p) { free(p); }

static inline void microkernel4x4(const double* __restrict__ A_tile,const double* __restrict__ B_tile,double* __restrict__ C_tile,int lda, int ldb, int ldc,int K)
{
    __m256d c0 = _mm256_loadu_pd(&C_tile[0 * ldc]);
    __m256d c1 = _mm256_loadu_pd(&C_tile[1 * ldc]);
    __m256d c2 = _mm256_loadu_pd(&C_tile[2 * ldc]);
    __m256d c3 = _mm256_loadu_pd(&C_tile[3 * ldc]);

    for (int k = 0; k < K; ++k) {
        __m256d bvec = _mm256_loadu_pd(&B_tile[k * ldb]);
        double a0 = A_tile[0 * lda + k];
        double a1 = A_tile[1 * lda + k];
        double a2 = A_tile[2 * lda + k];
        double a3 = A_tile[3 * lda + k];

        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a0), bvec, c0);
        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a1), bvec, c1);
        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a2), bvec, c2);
        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a3), bvec, c3);
    }

    _mm256_storeu_pd(&C_tile[0 * ldc], c0);
    _mm256_storeu_pd(&C_tile[1 * ldc], c1);
    _mm256_storeu_pd(&C_tile[2 * ldc], c2);
    _mm256_storeu_pd(&C_tile[3 * ldc], c3);
}

static inline void scalar_small(const double* __restrict__ A_tile, const double* __restrict__ B_tile,double* __restrict__ C_tile,int lda, int ldb, int ldc, int K, int M, int N)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += A_tile[i * lda + k] * B_tile[k * ldb + j];
            C_tile[i * ldc + j] += acc;
        }
}

void gemm_tiled_4x4(const double* A, const double* B, double* C,int N, int MC = 256, int KC = 128, int NC = 256,int lda = 0, int ldb = 0, int ldc = 0)
{
    if (!lda) lda = N;
    if (!ldb) ldb = N;
    if (!ldc) ldc = N;

    // dynamic
    #pragma omp parallel for collapse(2) schedule(static)
    for (int jc = 0; jc < N; jc += NC) {
        for (int ic = 0; ic < N; ic += MC) {
            for (int pc = 0; pc < N; pc += KC) {
                int jMax = min(jc + NC, N);
                int iMax = min(ic + MC, N);
                int pMax = min(pc + KC, N);
                int klen = pMax - pc;

                for (int j = jc; j < jMax; j += NR) {
                    int j_rem = min(NR, jMax - j);
                    for (int i = ic; i < iMax; i += MR) {
                        int i_rem = min(MR, iMax - i);

                        double* Cblk = &C[i * ldc + j];
                        const double* Ablk = &A[i * lda + pc];
                        const double* Bblk = &B[pc * ldb + j];

                        if (i_rem == MR && j_rem == NR)
                            microkernel4x4(Ablk, Bblk, Cblk, lda, ldb, ldc, klen);
                        else
                            scalar_small(Ablk, Bblk, Cblk, lda, ldb, ldc, klen, i_rem, j_rem);
                    }
                }
            }
        }
    }
}

// ======================================================
// ==================== MAIN ============================
// ======================================================
int main(int argc, char** argv)
{
    check_simd_support(); // ← Detect SIMD capabilities

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads [MC KC NC]\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    int MC = (argc > 3) ? atoi(argv[3]) : 256;
    int KC = (argc > 4) ? atoi(argv[4]) : 128;
    int NC = (argc > 5) ? atoi(argv[5]) : 256;

    omp_set_dynamic(0);
    omp_set_num_threads(T);

    size_t ne = (size_t)N * (size_t)N;
    double *A = (double*)aligned_malloc(sizeof(double) * ne);
    double *B = (double*)aligned_malloc(sizeof(double) * ne);
    double *C = (double*)aligned_malloc(sizeof(double) * ne);

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i=0; i<N*N; i++) { A[i]=dist(rng); B[i]=dist(rng); C[i]=0.0; }

    cout << fixed << setprecision(6);
    cout << "N="<<N<<" threads="<<T<<" MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";

    double t0 = omp_get_wtime();
    gemm_tiled_4x4(A, B, C, N, MC, KC, NC);
    double t1 = omp_get_wtime();

    double elapsed = t1 - t0;
    long double checksum = 0.0L;
    #pragma omp parallel for reduction(+ : checksum)
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            checksum += C[i*N + j];

    double gflops = (2.0 * N * N * N) / (elapsed * 1e9);
    cout << "Elapsed(s): " << elapsed << "  GFLOPs: " << gflops
         << "  checksum: " << setprecision(2) << (double)checksum << "\n";

    // aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}