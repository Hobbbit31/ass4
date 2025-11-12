// gemm_packed.cpp
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

// microkernel: A_tile: (MR x K) with lda, B_tile is packed with ldb_pack, C_tile has ldc
static inline void microkernel4x4_packed(const double* __restrict__ A_tile,
                                         const double* __restrict__ B_packed,
                                         double* __restrict__ C_tile,
                                         int lda, int ldb_pack, int ldc, int K)
{
    // load C (we may still need unaligned loads/stores for C if ldc not aligned)
    __m256d c0 = _mm256_loadu_pd(&C_tile[0 * ldc]);
    __m256d c1 = _mm256_loadu_pd(&C_tile[1 * ldc]);
    __m256d c2 = _mm256_loadu_pd(&C_tile[2 * ldc]);
    __m256d c3 = _mm256_loadu_pd(&C_tile[3 * ldc]);

    for (int k = 0; k < K; ++k) {
        // B_packed layout: row-major by k, with leading dimension ldb_pack (== blockJ)
        // we load 4 contiguous doubles (aligned since packed buffer is aligned)
        const double* brow = &B_packed[k * ldb_pack];
        __m256d bvec = _mm256_load_pd(brow); // packed buffer is 32/64B aligned

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

static inline void scalar_small(const double* __restrict__ A_tile, const double* __restrict__ B_tile,
                                double* __restrict__ C_tile, int lda, int ldb, int ldc,
                                int K, int M, int N)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += A_tile[i * lda + k] * B_tile[k * ldb + j];
            C_tile[i * ldc + j] += acc;
        }
}

void gemm_tiled_packed(const double* A, const double* B, double* C, int N,
                       int MC = 256, int KC = 128, int NC = 256,
                       int lda = 0, int ldb = 0, int ldc = 0)
{
    if (!lda) lda = N;
    if (!ldb) ldb = N;
    if (!ldc) ldc = N;

    // Outer parallelization over jc so each thread gets its own packed buffer safely.
    // This reduces contention and makes packing per-thread.
    #pragma omp parallel
    {
        // set up per-thread packed buffer maximum size (KC x NC)
        int maxKC = KC;
        int maxNC = NC;
        size_t maxBytes = (size_t)maxKC * (size_t)maxNC * sizeof(double);
        double* B_packed = (double*)aligned_malloc(maxBytes, 64);
        if (!B_packed) {
            fprintf(stderr, "B_packed allocation failed\n");
            omp_set_lock(nullptr); // noop to avoid warnings
        }

        #pragma omp for schedule(static)
        for (int jc = 0; jc < N; jc += NC) {
            int jMax = min(jc + NC, N);
            int blockJ_total = jMax - jc;

            for (int ic = 0; ic < N; ic += MC) {
                int iMax = min(ic + MC, N);

                for (int pc = 0; pc < N; pc += KC) {
                    int pMax = min(pc + KC, N);
                    int klen = pMax - pc;

                    // Pack B block: size = klen x blockJ_total
                    // layout: row-major by k: B_packed[k * blockJ_total + j]
                    for (int k = 0; k < klen; ++k) {
                        const double* Brow = &B[(pc + k) * ldb + jc];
                        double* packRow = &B_packed[k * blockJ_total];
                        // copy contiguous row of length blockJ_total
                        memcpy(packRow, Brow, sizeof(double) * blockJ_total);
                    }

                    // Inner compute over small tiles using packed B
                    for (int j = 0; j < blockJ_total; j += NR) {
                        int j_rem = min(NR, blockJ_total - j);
                        for (int i = ic; i < iMax; i += MR) {
                            int i_rem = min(MR, iMax - i);

                            double* Cblk = &C[i * ldc + (jc + j)];
                            const double* Ablk = &A[i * lda + pc];
                            const double* Bblk_packed = &B_packed[0 * blockJ_total + j]; // offset j within packed block

                            if (i_rem == MR && j_rem == NR) {
                                // call vector microkernel with B_packed ld = blockJ_total
                                microkernel4x4_packed(Ablk, Bblk_packed, Cblk, lda, blockJ_total, ldc, klen);
                            } else {
                                // for edge cases, construct a small B_tile view using B_packed
                                // We can call scalar_small with ldb = blockJ_total
                                scalar_small(Ablk, Bblk_packed, Cblk, lda, blockJ_total, ldc, klen, i_rem, j_rem);
                            }
                        }
                    }
                } // pc
            } // ic
        } // jc

        if (B_packed) aligned_free(B_packed);
    } // omp parallel
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

    // Bind/places for OpenMP - also export these in shell if you prefer
    setenv("OMP_PROC_BIND","close",1);
    setenv("OMP_PLACES","cores",1);

    size_t ne = (size_t)N * (size_t)N;
    double *A = (double*)aligned_malloc(sizeof(double) * ne, 64);
    double *B = (double*)aligned_malloc(sizeof(double) * ne, 64);
    double *C = (double*)aligned_malloc(sizeof(double) * ne, 64);

    if (!A || !B || !C) {
        cerr << "Allocation failed\n";
        return 1;
    }

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i=0; i<ne; ++i) { A[i]=dist(rng); B[i]=dist(rng); C[i]=0.0; }

    cout << fixed << setprecision(6);
    cout << "N="<<N<<" threads="<<T<<" MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";

    // // Warm-up run (stabilize caches, prefetcher, thread placement)
    // gemm_tiled_packed(A, B, C, N, MC, KC, NC);

    // clear output matrix before timed run
    memset(C, 0, ne * sizeof(double));

    double t0 = omp_get_wtime();
    gemm_tiled_packed(A, B, C, N, MC, KC, NC);
    double t1 = omp_get_wtime();

    double elapsed = t1 - t0;
    long double checksum = 0.0L;
    #pragma omp parallel for reduction(+ : checksum) schedule(static)
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            checksum += C[i*N + j];

    double gflops = (2.0 * N * N * N) / (elapsed * 1e9);
    cout << "Elapsed(s): " << elapsed << "  GFLOPs: " << gflops
         << "  checksum: " << setprecision(2) << (double)checksum << "\n";

    aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}



// updated code which does the computation 
// gemm_packed_ab.cpp
#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
using namespace std;

// ======================================================
// SIMD FEATURE CHECK (same as before)
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
    cout << "Detected SIMD support → SSE="<<sse<<" AVX="<<avx<<" AVX2="<<avx2<<" AVX512="<<avx512<<"\n";
}

// ======================================================
// GEMM params
// ======================================================
constexpr int MR = 4;
constexpr int NR = 4;

static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
}
static inline void aligned_free(void* p) { free(p); }

// ======================================================
// microkernel reading A_packed (MR x K contiguous per-row) and B_packed (K x blockJ contiguous by row)
// A_packed layout: row-major per i: A_packed[i * K + k]
// B_packed layout: row-major per k: B_packed[k * blockJ + j]
// ======================================================
static inline void microkernel4x4_packedAB(const double* __restrict__ A_packed,
                                           const double* __restrict__ B_packed,
                                           double* __restrict__ C_tile,
                                           int lda_pack /*= K*/, int ldb_pack /*= blockJ*/, int ldc, int K)
{
    // load C (we use unaligned loads for C since ldc may not be packed-aligned)
    __m256d c0 = _mm256_loadu_pd(&C_tile[0 * ldc]);
    __m256d c1 = _mm256_loadu_pd(&C_tile[1 * ldc]);
    __m256d c2 = _mm256_loadu_pd(&C_tile[2 * ldc]);
    __m256d c3 = _mm256_loadu_pd(&C_tile[3 * ldc]);

    for (int k = 0; k < K; ++k) {
        // B_packed row k: contiguous of length ldb_pack; we will load 4 doubles starting at j offset in caller
        const double* Brow = &B_packed[k * ldb_pack];
        __m256d bvec = _mm256_load_pd(Brow); // requires B_packed alignment of 32 bytes (we guarantee it)

        // A_packed rows: each row length is lda_pack (== K)
        double a0 = A_packed[0 * lda_pack + k];
        double a1 = A_packed[1 * lda_pack + k];
        double a2 = A_packed[2 * lda_pack + k];
        double a3 = A_packed[3 * lda_pack + k];

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

// scalar fallback (unchanged signature but B_tile here can be packed with ldb_pack)
static inline void scalar_small(const double* __restrict__ A_tile, const double* __restrict__ B_tile,
                                double* __restrict__ C_tile, int lda, int ldb, int ldc,
                                int K, int M, int N)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += A_tile[i * lda + k] * B_tile[k * ldb + j];
            C_tile[i * ldc + j] += acc;
        }
}

// ======================================================
// GEMM with A and B packing (per-thread buffers)
// ======================================================
void gemm_tiled_packedAB(const double* A, const double* B, double* C, int N,
                         int MC = 256, int KC = 128, int NC = 256,
                         int lda = 0, int ldb = 0, int ldc = 0)
{
    if (!lda) lda = N;
    if (!ldb) ldb = N;
    if (!ldc) ldc = N;

    // parallel region: allocate per-thread packing buffers
    #pragma omp parallel
    {
        int maxMC = MC;
        int maxKC = KC;
        int maxNC = NC;

        size_t bytesA = (size_t)maxMC * (size_t)maxKC * sizeof(double); // A_packed: MC x KC
        size_t bytesB = (size_t)maxKC * (size_t)maxNC * sizeof(double); // B_packed: KC x NC

        double* A_packed = (double*)aligned_malloc(bytesA ? bytesA : 1, 64);
        double* B_packed = (double*)aligned_malloc(bytesB ? bytesB : 1, 64);
        if (!A_packed || !B_packed) {
            fprintf(stderr, "Packing buffer allocation failed\n");
            // fallback: avoid crash, but performance will be poor. We can still attempt direct compute.
        }

        #pragma omp for schedule(static)
        for (int jc = 0; jc < N; jc += NC) {
            int jMax = min(jc + NC, N);
            int blockJ_total = jMax - jc;

            for (int ic = 0; ic < N; ic += MC) {
                int iMax = min(ic + MC, N);
                int blockI_total = iMax - ic;

                for (int pc = 0; pc < N; pc += KC) {
                    int pMax = min(pc + KC, N);
                    int klen = pMax - pc;

                    // ----- Pack B: size = klen x blockJ_total -----
                    // layout: B_packed[k * blockJ_total + j]
                    for (int k = 0; k < klen; ++k) {
                        const double* Brow = &B[(pc + k) * ldb + jc];
                        double* packRow = &B_packed[k * blockJ_total];
                        memcpy(packRow, Brow, sizeof(double) * blockJ_total);
                    }

                    // ----- Pack A: size = blockI_total x klen -----
                    // layout: A_packed[i_local * klen + k] where i_local = i - ic
                    for (int i_local = 0; i_local < blockI_total; ++i_local) {
                        const double* Arow = &A[(ic + i_local) * lda + pc];
                        double* packRow = &A_packed[i_local * klen];
                        memcpy(packRow, Arow, sizeof(double) * klen);
                    }

                    // ---- compute over small tiles using packed A and B ----
                    for (int j = 0; j < blockJ_total; j += NR) {
                        int j_rem = min(NR, blockJ_total - j);
                        for (int i_local = 0; i_local < blockI_total; i_local += MR) {
                            int i_rem = min(MR, blockI_total - i_local);

                            double* Cblk = &C[(ic + i_local) * ldc + (jc + j)];
                            const double* Ablk_packed = &A_packed[i_local * klen];   // lda_pack = klen
                            const double* Bblk_packed = &B_packed[0 * blockJ_total + j]; // offset j within each B row

                            if (i_rem == MR && j_rem == NR) {
                                // call optimized microkernel
                                microkernel4x4_packedAB(Ablk_packed, Bblk_packed, Cblk, klen, blockJ_total, ldc, klen);
                            } else {
                                // fallback: use scalar_small with packed layout (ldb = blockJ_total)
                                scalar_small(Ablk_packed, Bblk_packed, Cblk, klen, blockJ_total, ldc, klen, i_rem, j_rem);
                            }
                        }
                    }
                } // pc
            } // ic
        } // jc

        if (A_packed) aligned_free(A_packed);
        if (B_packed) aligned_free(B_packed);
    } // omp parallel
}

// ======================================================
// MAIN
// ======================================================
int main(int argc, char** argv)
{
    check_simd_support();

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

    // bind OpenMP threads to cores (also possible via shell)
    setenv("OMP_PROC_BIND","close",1);
    setenv("OMP_PLACES","cores",1);

    size_t ne = (size_t)N * (size_t)N;
    double *A = (double*)aligned_malloc(sizeof(double) * ne, 64);
    double *B = (double*)aligned_malloc(sizeof(double) * ne, 64);
    double *C = (double*)aligned_malloc(sizeof(double) * ne, 64);
    if (!A || !B || !C) {
        cerr << "Allocation failed\n";
        return 1;
    }

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i=0; i<ne; ++i) { A[i]=dist(rng); B[i]=dist(rng); C[i]=0.0; }

    cout << fixed << setprecision(6);
    cout << "N="<<N<<" threads="<<T<<" MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";

    // warm-up to stabilize caches / prefetchers / thread placement
    gemm_tiled_packedAB(A, B, C, N, MC, KC, NC);

    // clear C then timed run
    memset(C, 0, ne * sizeof(double));
    double t0 = omp_get_wtime();
    gemm_tiled_packedAB(A, B, C, N, MC, KC, NC);
    double t1 = omp_get_wtime();

    double elapsed = t1 - t0;
    long double checksum = 0.0L;
    #pragma omp parallel for reduction(+ : checksum) schedule(static)
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            checksum += C[i*N + j];

    double gflops = (2.0 * N * N * N) / (elapsed * 1e9);
    cout << "Elapsed(s): " << elapsed << "  GFLOPs: " << gflops
         << "  checksum: " << setprecision(2) << (double)checksum << "\n";

    aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}


////
////
////
///
////
////
// gemm_autotune.cpp
#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <signal.h>
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

    #pragma omp parallel for collapse(2) schedule(dynamic)
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

// ======================= AUTOTUNER HELPERS =========================
static double run_one(const double* A, const double* B, double* C, int N, int MC, int KC, int NC, int repeats)
{
    vector<double> gflops_list;
    for (int r=0;r<repeats;++r) {
        // reset C
        #pragma omp parallel for
        for (int i=0;i<N*N;++i) C[i]=0.0;

        double t0 = omp_get_wtime();
        gemm_tiled_4x4(A,B,C,N,MC,KC,NC);
        double t1 = omp_get_wtime();
        double elapsed = t1 - t0;
        double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

        // Compute checksum for validation
        long double checksum = 0.0L;
        #pragma omp parallel for reduction(+ : checksum)
        for (int i=0;i<N;i++)
            for (int j=0;j<N;j++)
                checksum += C[i*N + j];

        cout << "Run " << (r+1)
             << ": Time(s)=" << elapsed
             << "  GFLOPs=" << gflops
             << "  Checksum=" << (double)checksum
             << "\n";

        gflops_list.push_back(gflops);
    }
    sort(gflops_list.begin(), gflops_list.end());
    return gflops_list[gflops_list.size()/2]; // median
}

int main(int argc, char** argv)
{
    check_simd_support(); // show SIMD

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads [mode]\n";
        cerr << "mode = tune  -> runs autotune over a grid\n";
        cerr << "mode = run   -> runs single config: provide MC KC NC as next args\n";
        cerr << "Examples:\n";
        cerr << "  " << argv[0] << " 2048 8 tune\n";
        cerr << "  " << argv[0] << " 2048 8 run 256 128 4096\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    string mode = (argc>3) ? string(argv[3]) : string("tune");

    omp_set_dynamic(0);
    omp_set_num_threads(T);

    size_t ne = (size_t)N * (size_t)N;
    double *A = (double*)aligned_malloc(sizeof(double) * ne);
    double *B = (double*)aligned_malloc(sizeof(double) * ne);
    double *C = (double*)aligned_malloc(sizeof(double) * ne);
    if (!A || !B || !C) { cerr<<"Allocation failed\n"; return 1; }

    // initialize matrices once (same seed every run for reproducibility)
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i=0;i<ne;i++) { A[i]=dist(rng); B[i]=dist(rng); C[i]=0.0; }

    cout << fixed << setprecision(6);
    cout << "N="<<N<<" threads="<<T<<"\n";

    if (mode == "run") {
        if (argc < 7) { cerr<<"run mode requires MC KC NC\n"; return 1; }
        int MC = atoi(argv[4]);
        int KC = atoi(argv[5]);
        int NC = atoi(argv[6]);
        cout<<"Running single config MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";
        double g = run_one(A,B,C,N,MC,KC,NC,3);
        cout<<"Median GFLOPs: "<<g<<"\n";
    } else {
        // autotune grid - tuned for your reported caches (small grid)
        vector<int> KCS = {32,64,128,192,256};
        vector<int> MCS = {64,128,256,512};
        vector<int> NCS = {128,256,512,1024,2048,4096};
        int repeats = 1;

        double best_g = 0.0;
        array<int,3> best_cfg = {0,0,0};

        // header for CSV
        cout << "MC,KC,NC,median_GFLOPs\n";
        for (int kc: KCS) {
            for (int mc: MCS) {
                for (int nc: NCS) {
                    if (nc > N) continue; // skip too-large NC for this problem
                    if (mc > N) continue;
                    // ensure multiples of micro-kernel
                    if (mc % MR != 0 || nc % NR != 0) continue;
                    double g = run_one(A,B,C,N,mc,kc,nc,repeats);
                    cout << mc << "," << kc << "," << nc << "," << g << "\n";
                    if (g > best_g) { best_g = g; best_cfg = {mc,kc,nc}; }
                }
            }
        }

        cout << "BEST: MC="<<best_cfg[0]<<" KC="<<best_cfg[1]<<" NC="<<best_cfg[2]<<" GFLOPs="<<best_g<<"Time="<<"\n";
    }

    aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}


///
//
//
//
//
//
//
//
//
//
//
//
//
// gemm_autotune.cpp
#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <signal.h>
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

    #pragma omp parallel for collapse(2) schedule(dynamic)
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

// ======================= AUTOTUNER HELPERS =========================
static double run_one(const double* A, const double* B, double* C, int N, int MC, int KC, int NC, int repeats)
{
    vector<double> gflops_list;
    for (int r=0;r<repeats;++r) {
        // reset C
        #pragma omp parallel for
        for (int i=0;i<N*N;++i) C[i]=0.0;

        double t0 = omp_get_wtime();
        gemm_tiled_4x4(A,B,C,N,MC,KC,NC);
        double t1 = omp_get_wtime();
        double elapsed = t1 - t0;
        double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

        // Compute checksum for validation
        long double checksum = 0.0L;
        #pragma omp parallel for reduction(+ : checksum)
        for (int i=0;i<N;i++)
            for (int j=0;j<N;j++)
                checksum += C[i*N + j];

        cout << "Run " << (r+1)
             << ": Time(s)=" << elapsed
             << "  GFLOPs=" << gflops
             << "  Checksum=" << (double)checksum
             << "\n";

        gflops_list.push_back(gflops);
    }
    sort(gflops_list.begin(), gflops_list.end());
    return gflops_list[gflops_list.size()/2]; // median
}

int main(int argc, char** argv)
{
    check_simd_support(); // show SIMD

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads [mode]\n";
        cerr << "mode = tune  -> runs autotune over a grid\n";
        cerr << "mode = run   -> runs single config: provide MC KC NC as next args\n";
        cerr << "Examples:\n";
        cerr << "  " << argv[0] << " 2048 8 tune\n";
        cerr << "  " << argv[0] << " 2048 8 run 256 128 4096\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    string mode = (argc>3) ? string(argv[3]) : string("tune");

    omp_set_dynamic(0);
    omp_set_num_threads(T);

    size_t ne = (size_t)N * (size_t)N;
    double *A = (double*)aligned_malloc(sizeof(double) * ne);
    double *B = (double*)aligned_malloc(sizeof(double) * ne);
    double *C = (double*)aligned_malloc(sizeof(double) * ne);
    if (!A || !B || !C) { cerr<<"Allocation failed\n"; return 1; }

    // initialize matrices once (same seed every run for reproducibility)
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i=0;i<ne;i++) { A[i]=dist(rng); B[i]=dist(rng); C[i]=0.0; }

    cout << fixed << setprecision(6);
    cout << "N="<<N<<" threads="<<T<<"\n";

    if (mode == "run") {
        if (argc < 7) { cerr<<"run mode requires MC KC NC\n"; return 1; }
        int MC = atoi(argv[4]);
        int KC = atoi(argv[5]);
        int NC = atoi(argv[6]);
        cout<<"Running single config MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";
        double g = run_one(A,B,C,N,MC,KC,NC,3);
        cout<<"Median GFLOPs: "<<g<<"\n";
    } else {
        // autotune grid - tuned for your reported caches (small grid)
        vector<int> KCS = {32,64,128,192,256};
        vector<int> MCS = {64,128,256,512};
        vector<int> NCS = {128,256,512,1024,2048,4096};
        int repeats = 1;

        double best_g = 0.0;
        array<int,3> best_cfg = {0,0,0};

        // header for CSV
        cout << "MC,KC,NC,median_GFLOPs\n";
        for (int kc: KCS) {
            for (int mc: MCS) {
                for (int nc: NCS) {
                    if (nc > N) continue; // skip too-large NC for this problem
                    if (mc > N) continue;
                    // ensure multiples of micro-kernel
                    if (mc % MR != 0 || nc % NR != 0) continue;
                    double g = run_one(A,B,C,N,mc,kc,nc,repeats);
                    cout << mc << "," << kc << "," << nc << "," << g << "\n";
                    if (g > best_g) { best_g = g; best_cfg = {mc,kc,nc}; }
                }
            }
        }

        cout << "BEST: MC="<<best_cfg[0]<<" KC="<<best_cfg[1]<<" NC="<<best_cfg[2]<<" GFLOPs="<<best_g<<"Time="<<"\n";
    }

    aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}


// 
// 
// 
// 
// 
// 
// /
// 
// 
// 
// 
// 
// 
// 
// 
#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <signal.h>
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

    #pragma omp parallel for collapse(2) schedule(dynamic)
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

// ======================= RUN HELPERS =========================
struct RunStats {
    double gflops;
    double elapsed;
};

static RunStats run_one(const double* A, const double* B, double* C, int N, int MC, int KC, int NC)
{
    #pragma omp parallel for
    for (int i=0;i<N*N;++i) C[i]=0.0;

    double t0 = omp_get_wtime();
    gemm_tiled_4x4(A,B,C,N,MC,KC,NC);
    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;
    double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

    long double checksum = 0.0L;
    #pragma omp parallel for reduction(+ : checksum)
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            checksum += C[i*N + j];

    cout << "Time(s)=" << elapsed << "  GFLOPs=" << gflops
         << "  Checksum=" << (double)checksum << "\n";

    return {gflops, elapsed};
}

// ======================================================
// ==================== MAIN ============================
// ======================================================
int main(int argc, char** argv)
{
    check_simd_support(); // show SIMD

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads MC KC NC\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    int MC = 64;
    int KC = 32;
    int NC = 128;

    omp_set_dynamic(0);
    omp_set_num_threads(T);

    size_t ne = (size_t)N * (size_t)N;
    double *A = (double*)aligned_malloc(sizeof(double) * ne);
    double *B = (double*)aligned_malloc(sizeof(double) * ne);
    double *C = (double*)aligned_malloc(sizeof(double) * ne);
    if (!A || !B || !C) { cerr<<"Allocation failed\n"; return 1; }

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i=0;i<ne;i++) { A[i]=dist(rng); B[i]=dist(rng); }

    cout << fixed << setprecision(2);
    cout << "N="<<N<<" threads="<<T<<"\n";

    cout <<"Running single config MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";
    RunStats rs = run_one(A,B,C,N,MC,KC,NC);
    cout<<"GFLOPs="<<rs.gflops<<"  Time(s)="<<rs.elapsed<<"\n";

    // aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}
