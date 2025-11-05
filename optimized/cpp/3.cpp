// transpose/gemm_transpose.cpp
// Usage: ./gemm_transpose N num_threads
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

static inline void transpose_B(const double* B, double* Bt, int N){
    #pragma omp parallel for schedule(static)
    for(int k=0;k<N;++k){
        const double* bcol = B + k;
        double* btrow = Bt + (size_t)k*N;
        for(int j=0;j<N;++j) btrow[j] = bcol[(size_t)j*N];
    }
}

static inline void matmul_TB(const double* A, const double* Bt, double* C, int N){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i){
        double* crow = C + (size_t)i*N;
        for(int k=0;k<N;++k){
            double a = A[(size_t)i*N + k];
            const double* btrow = Bt + (size_t)k*N;
            #pragma omp simd
            for(int j=0;j<N;++j) crow[j] += a * btrow[j];
        }
    }
}

int main(int argc,char** argv){
    if(argc<3){ cerr<<"Usage: "<<argv[0]<<" N num_threads\n"; return 1; }
    int N=atoi(argv[1]); int T=atoi(argv[2]); omp_set_num_threads(T);
    vector<double> A((size_t)N*N), B((size_t)N*N), Bt((size_t)N*N), C((size_t)N*N,0.0);
    mt19937_64 rng(12345); normal_distribution<double> dist(0,1);
    for(size_t i=0;i<(size_t)N*N;i++){ A[i]=dist(rng); B[i]=dist(rng); }

    double t0=omp_get_wtime(); transpose_B(B.data(),Bt.data(),N); double t1=omp_get_wtime();
    double t2=omp_get_wtime(); matmul_TB(A.data(),Bt.data(),C.data(),N); double t3=omp_get_wtime();
    cout<<"N="<<N<<" T="<<T<<" transpose="<<(t1-t0)<<" s gemm="<<(t3-t2)<<" s total="<<(t3-t0)<<" s\n";
    long double s=0; for(size_t i=0;i<(size_t)N*N;i++) s+=C[i]; cout<<"checksum="<<(double)s<<"\n";
    return 0;
}

// Running optimized binary
// N=1024 T=4 transpose=0.00377916 s gemm=0.0624044 s total=0.0661836 s
// checksum=36995.5