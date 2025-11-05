# CXX = g++
# CXXFLAGS = -O3 -march=native -fopenmp -DNDEBUG
# LDFLAGS = -fopenmp

# all: optimized/gemm_opt

# optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
# 	mkdir -p optimized
# 	$(CXX) $(CXXFLAGS) -o optimized/gemm_opt optimized/cpp/gemm_opt.cpp $(LDFLAGS)

# clean:
# 	rm -f optimized/gemm_opt
CXX = g++
CXXFLAGS = -O3 -march=native -mtune=native -funroll-loops -fopenmp -DNDEBUG
LDFLAGS = -fopenmp -lnuma

all: optimized/gemm_opt

optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
	mkdir -p optimized
	$(CXX) $(CXXFLAGS) -o optimized/gemm_opt optimized/cpp/gemm_opt.cpp $(LDFLAGS)

clean:
	rm -f optimized/gemm_opt
