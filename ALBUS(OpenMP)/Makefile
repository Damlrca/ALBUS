all:spmv_albus

.PHONY : spmv_albus
#compilers(g++ or icc)

#CC=icpc
#CC = g++
CC = riscv64-unknown-linux-gnu-g++

#riscv64-unknown-linux-gnu-g++
spmv_albus:
	$(CC) -march=rv64gcv -opt-prefetch=3 -fopenmp -O3 -o spmv_albus main.cpp

#g++
#spmv_albus:
#	$(CC) -march=native -mavx -mavx2 -opt-prefetch=3 -fopenmp -O3 -o spmv_albus main.cpp

#icc
#spmv_albus:
#	$(CC) -g -w -fno-alias -qopt-report -qopt-report-phase=all -mbranches-within-32B-boundaries -falign-double \
#         -inline-calloc -falign-loops=8 -xCORE-AVX2 -fopenmp -O3 -o spmv_albus main.cpp

.PHONY:clean
	rm spmv_albus
