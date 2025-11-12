// gemm_autotune_avx512.cpp
#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <signal.h>
using namespace std;

// Runtime MR/NR (set after SIMD detection)
int MR_R = 4;
int NR_R = 4;

// SIMD feature flags (set by check_simd_support)
bool g_sse = false, g_avx = false, g_avx2 = false, g_avx512 = false;

// ======================================================
// =============== SIMD FEATURE CHECK ===================
// ======================================================
void check_simd_support() {
    unsigned a,b,c,d;
    g_sse = g_avx = g_avx2 = g_avx512 = false;

    __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(1));
    g_sse = (d >> 25) & 1;
    if ((c & (1<<27)) && (c & (1<<28))) {
        unsigned x0a,x0d;
        __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
        if ((x0a & 6) == 6) g_avx = true;
    }

    __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(7));
    if (g_avx && (b & (1<<5))) g_avx2 = true;
    if ((b & (1<<16))) {
        unsigned x0a,x0d;
        __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
        // check OPMASK(1<<5), ZMM state (bits 5..7) -> mask for AVX-512
        if ((x0a & 0xE0) == 0xE0) g_avx512 = true;
    }

    cout << "Detected SIMD support → ";
    cout << "SSE=" << g_sse << " AVX=" << g_avx << " AVX2=" << g_avx2 << " AVX512=" << g_avx512 << "\n";
    if (g_avx512) {
        cout << "→ Best available SIMD: AVX-512\n";
        MR_R = NR_R = 8;
    } else if (g_avx2) {
        cout << "→ Best available SIMD: AVX2\n";
        MR_R = NR_R = 4;
    } else if (g_avx) {
        cout << "→ Best available SIMD: AVX\n";
        MR_R = NR_R = 4;
    } else {
        cout << "→ Fallback: SSE or Scalar\n";
        MR_R = NR_R = 1;
    }
}

// ======================================================
// =============== GEMM IMPLEMENTATION ==================
// ======================================================

static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
}
static inline void aligned_free(void* p) { free(p); }

// AVX2 4x4 microkernel (uses __m256d)
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

// AVX-512 8x8 microkernel (uses __m512d)
#if defined(__AVX512F__) || defined(__AVX512F)
static inline void microkernel8x8(const double* __restrict__ A_tile,const double* __restrict__ B_tile,double* __restrict__ C_tile,int lda, int ldb, int ldc,int K)
{
    __m512d c0 = _mm512_loadu_pd(&C_tile[0 * ldc]);
    __m512d c1 = _mm512_loadu_pd(&C_tile[1 * ldc]);
    __m512d c2 = _mm512_loadu_pd(&C_tile[2 * ldc]);
    __m512d c3 = _mm512_loadu_pd(&C_tile[3 * ldc]);
    __m512d c4 = _mm512_loadu_pd(&C_tile[4 * ldc]);
    __m512d c5 = _mm512_loadu_pd(&C_tile[5 * ldc]);
    __m512d c6 = _mm512_loadu_pd(&C_tile[6 * ldc]);
    __m512d c7 = _mm512_loadu_pd(&C_tile[7 * ldc]);

    for (int k = 0; k < K; ++k) {
        __m512d bvec = _mm512_loadu_pd(&B_tile[k * ldb]);
        double a0 = A_tile[0 * lda + k];
        double a1 = A_tile[1 * lda + k];
        double a2 = A_tile[2 * lda + k];
        double a3 = A_tile[3 * lda + k];
        double a4 = A_tile[4 * lda + k];
        double a5 = A_tile[5 * lda + k];
        double a6 = A_tile[6 * lda + k];
        double a7 = A_tile[7 * lda + k];

        c0 = _mm512_fmadd_pd(_mm512_set1_pd(a0), bvec, c0);
        c1 = _mm512_fmadd_pd(_mm512_set1_pd(a1), bvec, c1);
        c2 = _mm512_fmadd_pd(_mm512_set1_pd(a2), bvec, c2);
        c3 = _mm512_fmadd_pd(_mm512_set1_pd(a3), bvec, c3);
        c4 = _mm512_fmadd_pd(_mm512_set1_pd(a4), bvec, c4);
        c5 = _mm512_fmadd_pd(_mm512_set1_pd(a5), bvec, c5);
        c6 = _mm512_fmadd_pd(_mm512_set1_pd(a6), bvec, c6);
        c7 = _mm512_fmadd_pd(_mm512_set1_pd(a7), bvec, c7);
    }

    _mm512_storeu_pd(&C_tile[0 * ldc], c0);
    _mm512_storeu_pd(&C_tile[1 * ldc], c1);
    _mm512_storeu_pd(&C_tile[2 * ldc], c2);
    _mm512_storeu_pd(&C_tile[3 * ldc], c3);
    _mm512_storeu_pd(&C_tile[4 * ldc], c4);
    _mm512_storeu_pd(&C_tile[5 * ldc], c5);
    _mm512_storeu_pd(&C_tile[6 * ldc], c6);
    _mm512_storeu_pd(&C_tile[7 * ldc], c7);
}
#endif

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

void gemm_tiled(const double* A, const double* B, double* C,int N, int MC = 256, int KC = 128, int NC = 256,int lda = 0, int ldb = 0, int ldc = 0)
{
    if (!lda) lda = N;
    if (!ldb) ldb = N;
    if (!ldc) ldc = N;

    int MR = MR_R;
    int NR = NR_R;

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

                        if (i_rem == MR && j_rem == NR) {
                            if (MR == 4 && NR == 4)
                                microkernel4x4(Ablk, Bblk, Cblk, lda, ldb, ldc, klen);
#if defined(__AVX512F__) || defined(__AVX512F)
                            else if (MR == 8 && NR == 8)
                                microkernel8x8(Ablk, Bblk, Cblk, lda, ldb, ldc, klen);
#endif
                            else
                                scalar_small(Ablk, Bblk, Cblk, lda, ldb, ldc, klen, i_rem, j_rem);
                        } else {
                            scalar_small(Ablk, Bblk, Cblk, lda, ldb, ldc, klen, i_rem, j_rem);
                        }
                    }
                }
            }
        }
    }
}

// ======================= AUTOTUNER HELPERS =========================
struct RunStats {
    double gflops;
    double elapsed;
};

static RunStats run_one(const double* A, const double* B, double* C, int N, int MC, int KC, int NC)
{
    #pragma omp parallel for
    for (int i=0;i<N*N;++i) C[i]=0.0;

    double t0 = omp_get_wtime();
    gemm_tiled(A,B,C,N,MC,KC,NC);
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
    check_simd_support(); // show SIMD and set MR_R/NR_R

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads [mode]\n";
        cerr << "mode = tune  -> runs autotune over a grid\n";
        cerr << "mode = run   -> runs single config: provide MC KC NC as next args\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    string mode = (argc>3) ? string(argv[3]) : string("tune");

    omp_set_dynamic(0);
    omp_set_num_threads(T);

    cout << "Using MR=" << MR_R << " NR=" << NR_R << "\n";

    size_t ne = (size_t)N * (size_t)N;
    double *A = (double*)aligned_malloc(sizeof(double) * ne);
    double *B = (double*)aligned_malloc(sizeof(double) * ne);
    double *C = (double*)aligned_malloc(sizeof(double) * ne);
    if (!A || !B || !C) { cerr<<"Allocation failed\n"; return 1; }

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i=0;i<ne;i++) { A[i]=dist(rng); B[i]=dist(rng); }

    cout << fixed << setprecision(6);
    cout << "N="<<N<<" threads="<<T<<"\n";

    if (mode == "run") {
        if (argc < 7) { cerr<<"run mode requires MC KC NC\n"; return 1; }
        int MC = atoi(argv[4]);
        int KC = atoi(argv[5]);
        int NC = atoi(argv[6]);
        cout<<"Running single config MC="<<MC<<" KC="<<KC<<" NC="<<NC<<"\n";
        RunStats rs = run_one(A,B,C,N,MC,KC,NC);
        cout<<"GFLOPs="<<rs.gflops<<"  Time(s)="<<rs.elapsed<<"\n";
    }
    else {  // autotune mode
        vector<int> KCS = {32,64,128,256};
        vector<int> MCS = {64,128,256,512,1024};
        vector<int> NCS = {128,256,512,1024,2048};

        double best_g = 0.0;
        double min_time = 1e9;
        array<int,3> best_g_cfg = {0,0,0};
        array<int,3> min_time_cfg = {0,0,0};

        auto tune_start = chrono::high_resolution_clock::now();
        cout << "MC,KC,NC,GFLOPs,Time(s)\n";

        for (int kc: KCS)
            for (int mc: MCS)
                for (int nc: NCS) {
                    if (mc > N || nc > N) continue;
                    if (mc % MR_R != 0 || nc % NR_R != 0) continue;

                    RunStats rs = run_one(A,B,C,N,mc,kc,nc);
                    cout << mc << "," << kc << "," << nc << ","
                         << rs.gflops << "," << rs.elapsed << "\n";

                    if (rs.gflops > best_g) {
                        best_g = rs.gflops;
                        best_g_cfg = {mc,kc,nc};
                    }
                    if (rs.elapsed < min_time) {
                        min_time = rs.elapsed;
                        min_time_cfg = {mc,kc,nc};
                    }
                }

        auto tune_end = chrono::high_resolution_clock::now();
        double total_time = chrono::duration<double>(tune_end - tune_start).count();

        cout << "=========================================\n";
        cout << "BEST GFLOPs CONFIG: MC="<<best_g_cfg[0]
             <<" KC="<<best_g_cfg[1]
             <<" NC="<<best_g_cfg[2]
             <<"  GFLOPs="<<best_g<<"\n";
        cout << "MIN TIME CONFIG:   MC="<<min_time_cfg[0]
             <<" KC="<<min_time_cfg[1]
             <<" NC="<<min_time_cfg[2]
             <<"  Time="<<min_time<<" seconds\n";
        cout << "TOTAL AUTOTUNE TIME: " << total_time << " seconds\n";
        cout << "=========================================\n";
    }

    aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}
