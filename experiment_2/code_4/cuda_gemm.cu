// cuda_gemm.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// CUDA核函数：执行矩阵乘法
__global__ void matrixMulCUDA(double* A, double* B, double* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 计算行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 计算列索引

    if(row < M && col < K){
        double sum = 0.0;
        for(int l = 0; l < N; l++){
            sum += A[row * N + l] * B[l * K + col];
        }
        C[row * K + col] = sum;
    }
}

// 主函数
int main(int argc, char* argv[]){
    int M, N, K;
    double *h_A, *h_B, *h_C;
    double *d_A, *d_B, *d_C;
    size_t size_A, size_B, size_C;
    double start_time, end_time;

    if(argc != 4){
        printf("Usage: %s M N K\n", argv[0]);
        printf("Matrix dimensions: A(MxN) * B(NxK) = C(MxK)\n");
        return 0;
    }

    // 读取矩阵维度
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    // 分配主机内存
    size_A = M * N * sizeof(double);
    size_B = N * K * sizeof(double);
    size_C = M * K * sizeof(double);
    h_A = (double*)malloc(size_A);
    h_B = (double*)malloc(size_B);
    h_C = (double*)malloc(size_C);

    // 初始化矩阵A和B
    srand(time(NULL));
    for(int i = 0; i < M*N; i++) h_A[i] = ((double)rand()) / RAND_MAX;
    for(int i = 0; i < N*K; i++) h_B[i] = ((double)rand()) / RAND_MAX;

    // 分配设备内存
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // 将矩阵A和B从主机复制到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 定义CUDA网格和块的维度
    dim3 dimBlock(16, 16); // 每个块16x16线程
    dim3 dimGrid((K + dimBlock.x -1)/dimBlock.x, (M + dimBlock.y -1)/dimBlock.y);

    // 同步所有设备线程
    cudaDeviceSynchronize();

    // 记录开始时间
    start_time = clock();

    // 启动CUDA核函数
    matrixMulCUDA<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // 等待CUDA核函数完成
    cudaDeviceSynchronize();

    // 记录结束时间
    end_time = clock();

    // 将结果矩阵C从设备复制到主机
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 计算时间差（以秒为单位）
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // 打印结果
    // 打印矩阵A、B和C（可选，适用于小规模矩阵）
    /*
    printf("Matrix A:\n");
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++) printf("%0.2f ", h_A[i*N + j]);
        printf("\n");
    }
    printf("\nMatrix B:\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < K; j++) printf("%0.2f ", h_B[i*K + j]);
        printf("\n");
    }
    printf("\nMatrix C:\n");
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++) printf("%0.2f ", h_C[i*K + j]);
        printf("\n");
    }
    printf("\n");
    */

    printf("CUDA Matrix multiplication completed in %f seconds.\n", elapsed_time);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}