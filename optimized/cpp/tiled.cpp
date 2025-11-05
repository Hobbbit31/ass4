// #include <bits/stdc++.h>
// #include <omp.h>
// using namespace std;

// // Tiled GEMM
// static inline void gemm_tiled(const double* A, const double* B, double* C,
//                               int N, int b)
// {
//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int ii = 0; ii < N; ii += b) {
//         for (int jj = 0; jj < N; jj += b) {
//             for (int kk = 0; kk < N; kk += b) {
//                 int iMax = min(ii + b, N);
//                 int jMax = min(jj + b, N);
//                 int kMax = min(kk + b, N);
//                 for (int i = ii; i < iMax; ++i) {
//                     double* crow = C + (size_t)i * N;
//                     for (int k = kk; k < kMax; ++k) {
//                         const double aik = A[(size_t)i * N + k];
//                         const double* brow = B + (size_t)k * N;
//                         #pragma omp simd
//                         for (int j = jj; j < jMax; ++j) {
//                             crow[j] += aik * brow[j];
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
//     if (N <= 0 || T <= 0) { cerr << "N and num_threads must be > 0\n"; return 1; }

//     omp_set_num_threads(T);

//     vector<double> A((size_t)N*N), B((size_t)N*N);
//     std::mt19937_64 rng(12345);
//     std::normal_distribution<double> dist(0.0, 1.0);
//     for (size_t i = 0; i < (size_t)N*N; ++i) { A[i] = dist(rng); B[i] = dist(rng); }

//     // Candidate tile sizes to test
//     vector<int> candidates = {64, 96, 128, 192, 256, 320, 384, 512};

//     double best_time = 1e100;
//     int best_b = -1;

//     for (int b : candidates) {
//         vector<double> C((size_t)N*N, 0.0);

//         double t0 = omp_get_wtime();
//         gemm_tiled(A.data(), B.data(), C.data(), N, b);
//         double t1 = omp_get_wtime();
//         double elapsed = t1 - t0;

//         long double s = 0.0L;
//         for (size_t i = 0; i < (size_t)N*N; ++i) s += C[i];

//         cout << "b=" << b << " time=" << elapsed << " s checksum="
//              << setprecision(17) << (double)s << "\n";

//         if (elapsed < best_time) {
//             best_time = elapsed;
//             best_b = b;
//         }
//     }

//     cout << "---- Best b=" << best_b << " time=" << best_time << " s ----\n";
//     return 0;
// }

// // b=64 time=51.6195 s checksum=-770903.52827476908
// // b=96 time=48.333963389000019 s checksum=-770903.52827476908
// // b=128 time=45.521458863000134 s checksum=-770903.52827476908
// // b=192 time=43.365270006000173 s checksum=-770903.52827476908
// // b=256 time=59.173056076999728 s checksum=-770903.52827476908
// // b=320 time=54.786918169999808 s checksum=-770903.52827476908
// // b=384 time=64.134110792999763 s checksum=-770903.52827476908
// // b=512 time=63.432770203000018 s checksum=-770903.52827476908
// // ---- Best b=192 time=43.365270006000173 s ----

// // b=64 time=35.8788 s checksum=-770903.52827476908
// // b=96 time=31.118608056000085 s checksum=-770903.52827476908
// // b=128 time=28.729585911000413 s checksum=-770903.52827476908
// // b=192 time=28.428354197000317 s checksum=-770903.52827476908
// // b=256 time=38.306936623999718 s checksum=-770903.52827476908
// // b=320 time=38.817141094999897 s checksum=-770903.52827476908
// // b=384 time=76.881791712999984 s checksum=-770903.52827476908
// // b=512 time=65.19470038999998 s checksum=-770903.52827476908
// // ---- Best b=192 time=28.428354197000317 s ----

// // Running baseline python implementation
// // N=10000 P=4 time=21.030031 seconds

// // Running baseline python implementation
// // N=10000 P=8 time=27.849910 seconds















#include <bits/stdc++.h>
#include <omp.h>
#include <immintrin.h>
using namespace std;

constexpr int MR = 4;  // microkernel rows
constexpr int NR = 4;  // microkernel cols

// AVX2 4x4 microkernel
static inline void microkernel4x4(const double* A, const double* B, double* C,
                                  int lda, int ldb, int ldc, int K) {
    __m256d c0 = _mm256_loadu_pd(&C[0*ldc]); 
    __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
    __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
    __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);

    for (int k = 0; k < K; ++k) {
        __m256d b = _mm256_loadu_pd(&B[k*ldb]); 
        __m256d a0 = _mm256_broadcast_sd(&A[0*lda + k]);
        __m256d a1 = _mm256_broadcast_sd(&A[1*lda + k]);
        __m256d a2 = _mm256_broadcast_sd(&A[2*lda + k]);
        __m256d a3 = _mm256_broadcast_sd(&A[3*lda + k]);

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

// Blocked GEMM
static void gemm_blocked(const double* A, const double* B, double* C, int N,
                         int MC, int KC, int NC) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int jc = 0; jc < N; jc += NC) {
        for (int pc = 0; pc < N; pc += KC) {
            for (int ic = 0; ic < N; ic += MC) {
                int jMax = min(jc + NC, N);
                int pMax = min(pc + KC, N);
                int iMax = min(ic + MC, N);

                for (int i = ic; i + MR <= iMax; i += MR) {
                    for (int j = jc; j + NR <= jMax; j += NR) {
                        double* Cblk = &C[i*N + j];
                        const double* Ablk = &A[i*N + pc];
                        const double* Bblk = &B[pc*N + j];
                        microkernel4x4(Ablk, Bblk, Cblk, N, N, N, pMax-pc);
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
    if (N <= 0 || T <= 0) { cerr << "N and num_threads must be > 0\n"; return 1; }

    omp_set_num_threads(T);

    vector<double> A((size_t)N*N), B((size_t)N*N), C((size_t)N*N, 0.0);
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < (size_t)N*N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // Candidate block sizes (MC, KC, NC)
    vector<int> MCs = {128, 192, 256};
    vector<int> KCs = {256, 512};
    vector<int> NCs = {1024, 2048, 4096};

    double best_time = 1e100;
    tuple<int,int,int> best_cfg;

    for (int MC : MCs) {
        for (int KC : KCs) {
            for (int NC : NCs) {
                fill(C.begin(), C.end(), 0.0);
                double t0 = omp_get_wtime();
                gemm_blocked(A.data(), B.data(), C.data(), N, MC, KC, NC);
                double t1 = omp_get_wtime();
                double elapsed = t1 - t0;
                long double s = 0.0L;
                for (size_t i = 0; i < (size_t)N*N; ++i) s += C[i];
                double gflops = (2.0 * N * N * N) / (elapsed * 1e9);
                cout << "MC="<<MC<<" KC="<<KC<<" NC="<<NC
                     <<" time="<<elapsed<<" s GFLOPs="<<gflops
                     <<" checksum="<<setprecision(17)<<(double)s<<"\n";

                if (elapsed < best_time) {
                    best_time = elapsed;
                    best_cfg = {MC, KC, NC};
                }
            }
        }
    }

    auto [bMC, bKC, bNC] = best_cfg;
    cout << "---- Best config: MC="<<bMC
         <<" KC="<<bKC<<" NC="<<bNC
         <<" time="<<best_time<<" s ----\n";
    return 0;
}
