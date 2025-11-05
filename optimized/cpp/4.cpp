// avx2/gemm_avx2.cpp
// Usage: ./gemm_avx2 N num_threads
#include <bits/stdc++.h>
#include <omp.h>
#include <immintrin.h>
using namespace std;

static void matmul_avx2(const double* A, const double* B, double* C, int N){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i){
        double* crow = C + (size_t)i*N;
        for(int k=0;k<N;++k){
            __m256d a4 = _mm256_broadcast_sd(&A[(size_t)i*N + k]);
            const double* brow = B + (size_t)k*N;
            int j=0;
            for(; j+3<N; j+=4){
                __m256d c = _mm256_loadu_pd(crow + j);
                __m256d b = _mm256_loadu_pd(brow + j);
                c = _mm256_fmadd_pd(a4, b, c);
                _mm256_storeu_pd(crow + j, c);
            }
            for(; j<N; ++j) crow[j] += A[(size_t)i*N + k] * brow[j];
        }
    }
}

int main(int argc,char** argv){
    if(argc<3){ cerr<<"Usage: "<<argv[0]<<" N num_threads\n"; return 1; }
    int N=atoi(argv[1]); int T=atoi(argv[2]); omp_set_num_threads(T);
    vector<double> A((size_t)N*N), B((size_t)N*N), C((size_t)N*N,0.0);
    mt19937_64 rng(12345); normal_distribution<double> dist(0,1);
    for(size_t i=0;i<(size_t)N*N;i++){ A[i]=dist(rng); B[i]=dist(rng); }
    double t0=omp_get_wtime(); matmul_avx2(A.data(),B.data(),C.data(),N); double t1=omp_get_wtime();
    cout<<"N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" s\n";
    long double s=0; for(size_t i=0;i<(size_t)N*N;i++) s+=C[i]; cout<<"checksum="<<(double)s<<"\n";
    return 0;
}

// Running optimized binary
// N=1024 T=4 time=0.0729237 s
// checksum=-30660.9