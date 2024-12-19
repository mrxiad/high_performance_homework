#!/bin/bash

# 执行 MPI 程序
echo "Running MPI program..."
./bin/mpi_hello

# 执行 OpenMP 程序
echo "Running OpenMP program..."
./bin/openmp_hello

# 执行 pthread 程序
echo "Running Pthreads program..."
./bin/pthread_hello

# 执行 CUDA 程序
echo "Running CUDA program..."
./bin/cuda_hello


echo "All programs executed."
