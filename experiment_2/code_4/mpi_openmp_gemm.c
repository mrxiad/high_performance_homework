// mpi_openmp_gemm.c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>

// 函数声明
void generate_matrix(double* mat, int rows, int cols);
void print_matrix(double* mat, int rows, int cols, const char* name);

int main(int argc, char* argv[]) {
    int rank, size;
    int M, N, K;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A, *local_C;
    int rows_per_proc;
    double start_time, end_time;

    MPI_Init(&argc, &argv);                       // 初始化MPI环境
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);         // 获取当前进程编号
    MPI_Comm_size(MPI_COMM_WORLD, &size);         // 获取进程总数

    if (argc != 4) {
        if(rank == 0){
            printf("Usage: mpirun -np <num_procs> %s M N K\n", argv[0]);
            printf("Matrix dimensions: A(MxN) * B(NxK) = C(MxK)\n");
        }
        MPI_Finalize();
        return 0;
    }

    // 读取矩阵维度
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    // 确保M能被进程数整除
    if(M % size != 0){
        if(rank == 0){
            printf("Error: M (%d) must be divisible by number of processes (%d).\n", M, size);
        }
        MPI_Finalize();
        return 0;
    }

    rows_per_proc = M / size;                     // 每个进程处理的行数

    // 分配本地A和C的存储空间
    local_A = (double*)malloc(rows_per_proc * N * sizeof(double));
    local_C = (double*)malloc(rows_per_proc * K * sizeof(double));

    // 初始化矩阵A和C（仅在rank 0中）
    if(rank == 0){
        A = (double*)malloc(M * N * sizeof(double));
        B = (double*)malloc(N * K * sizeof(double));
        C = (double*)malloc(M * K * sizeof(double));

        // 生成随机矩阵A和B
        srand(time(NULL));
        generate_matrix(A, M, N);
        generate_matrix(B, N, K);

        // 打印矩阵A和B（可选，适用于小规模矩阵）
        /*
        print_matrix(A, M, N, "A");
        print_matrix(B, N, K, "B");
        */
    }

    // 广播矩阵B到所有进程
    if(rank != 0){
        B = (double*)malloc(N * K * sizeof(double));
    }
    MPI_Bcast(B, N*K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 分发矩阵A的子块到各个进程
    MPI_Scatter(A, rows_per_proc*N, MPI_DOUBLE, local_A, rows_per_proc*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 同步所有进程，开始计时
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        start_time = MPI_Wtime();
    }

    // 并行执行本地的矩阵乘法
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < rows_per_proc; i++){
        for(int j = 0; j < K; j++){
            double sum = 0.0;
            for(int l = 0; l < N; l++){
                sum += local_A[i*N + l] * B[l*K + j];
            }
            local_C[i*K + j] = sum;
        }
    }

    // 收集所有进程的计算结果到矩阵C（仅在rank 0中）
    MPI_Gather(local_C, rows_per_proc*K, MPI_DOUBLE, C, rows_per_proc*K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 同步所有进程，结束计时
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        end_time = MPI_Wtime();
        // 打印矩阵C（可选，适用于小规模矩阵）
        /*
        print_matrix(C, M, K, "C");
        */
        printf("MPI+OpenMP Matrix multiplication completed in %f seconds.\n", end_time - start_time);
    }

    // 释放内存
    if(rank == 0){
        free(A);
        free(B);
        free(C);
    }
    free(local_A);
    free(local_C);
    if(rank != 0){
        free(B);
    }

    MPI_Finalize();                                // 结束MPI环境
    return 0;
}

// 生成随机矩阵
void generate_matrix(double* mat, int rows, int cols){
    for(int i = 0; i < rows*cols; i++){
        mat[i] = ((double)rand()) / RAND_MAX;
    }
}

// 打印矩阵（用于调试）
void print_matrix(double* mat, int rows, int cols, const char* name){
    printf("Matrix %s:\n", name);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%0.2f ", mat[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}