# CXX = g++
# CXXFLAGS = -O3 -march=native -fopenmp -DNDEBUG
# LDFLAGS = -fopenmp

# all: optimized/gemm_opt

# optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
# 	mkdir -p optimized
# 	$(CXX) $(CXXFLAGS) -o optimized/gemm_opt optimized/cpp/gemm_opt.cpp $(LDFLAGS)

# clean:
# 	rm -f optimized/gemm_opt
# CXX = g++
# CXXFLAGS = -O5 -march=native -mtune=native -funroll-loops -DNDEBUG -mavx512f -mavx512dq mavx2 -mfma 
# LDFLAGS = -fopenmp

# all: optimized/gemm_opt

# optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
# 	mkdir -p optimized
# 	$(CXX) $(CXXFLAGS) -o optimized/gemm_opt optimized/cpp/gemm_opt.cpp $(LDFLAGS)

# clean:
# 	rm -f optimized/gemm_opt


CXX = g++
# -O5 doesn’t exist (highest is -O3); also fix missing dash before -mavx2 -mavx512f -mavx512dq -DNDEBUG 
CXXFLAGS = -O3 -march=native -mtune=native -funroll-loops -mavx2 -mfma 
LDFLAGS = -fopenmp

all: optimized/gemm_opt

optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
	mkdir -p optimized
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f optimized/gemm_opt

