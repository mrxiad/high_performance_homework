#!/bin/bash

# 检查并创建输出目录
mkdir -p bin

# 编译 MPI 程序
echo "Compiling MPI program..."
mpicc -O3 -Wall mpi_hello.c -o ./bin/mpi_hello

# 编译 OpenMP 程序
echo "Compiling OpenMP program..."
gcc -O3 -Wall -fopenmp openmp_hello.c -o ./bin/openmp_hello

# 编译 pthread 程序
echo "Compiling Pthreads program..."
gcc -O3 -Wall pthread_hello.c -o ./bin/pthread_hello -lpthread

# 编译 CUDA 程序
echo "Compiling CUDA program..."
nvcc -O3 cuda_hello.cu -o ./bin/cuda_hello

echo "Build complete. All executables are in the bin/ directory."

# 完成后自动运行所有程序
# echo "Running all programs..."

# 运行所有并行程序
# echo "Running MPI program..."
# ./bin/mpi_hello
# echo "Running OpenMP program..."
# ./bin/openmp_hello
# echo "Running Pthreads program..."
# ./bin/pthread_hello
# echo "Running CUDA program..."
# ./bin/cuda_hello

# echo "All programs executed."
