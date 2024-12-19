// send_lower_triangle.c
#include <stdio.h>      // 标准输入输出库
#include <stdlib.h>     // 标准库
#include <mpi.h>        // MPI库

// 函数声明：打印矩阵（用于调试）
void print_matrix(double* A, int n, const char* name) {
    printf("Matrix %s:\n", name);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%6.2f ", A[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    int rank, size;    // 进程编号和总进程数
    int n = 4;         // 矩阵大小（n x n），可以根据需要调整

    MPI_Init(&argc, &argv);                      // 初始化MPI环境
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);        // 获取当前进程编号
    MPI_Comm_size(MPI_COMM_WORLD, &size);        // 获取进程总数

    if(size < 2) {
        if(rank == 0) {
            printf("请至少使用两个进程运行此程序。\n");
        }
        MPI_Finalize();
        return 0;
    }

    // 仅在发送进程（rank 0）中初始化矩阵A
    double* A = NULL;
    if(rank == 0) {
        A = (double*)malloc(n * n * sizeof(double));
        // 初始化矩阵A，例如A[i][j] = i * n + j + 1
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                A[i*n + j] = (double)(i * n + j + 1);
            }
        }
        // 打印初始化后的矩阵A
        print_matrix(A, n, "A");
    }

    // 定义自定义MPI数据类型来描述下三角部分
    MPI_Datatype lower_triangle_type;
    int count = 0;                    // 下三角元素总数
    for(int i = 0; i < n; i++) {
        count += (i + 1);             // 每行i有i+1个元素满足i >= j
    }

    // 分配一个临时数组来存储下三角元素
    double* lower_triangle = NULL;
    if(rank == 0) {
        lower_triangle = (double*)malloc(count * sizeof(double));
        int idx = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j <= i; j++) {
                lower_triangle[idx++] = A[i*n + j];
            }
        }
        // 打印下三角部分
        printf("Lower Triangle Elements to Send:\n");
        for(int i = 0; i < count; i++) {
            printf("%6.2f ", lower_triangle[i]);
            if((i+1) % (n) == 0) printf("\n");
        }
        printf("\n");
    }

    // 创建MPI自定义数据类型
    // 在这个例子中，我们选择使用MPI_DOUBLE的连续块，因为下三角部分在发送前已经被提取到一个连续数组
    // 因此，自定义数据类型并不复杂，但在更复杂的情况下，可能需要定义更复杂的类型
    MPI_Type_contiguous(count, MPI_DOUBLE, &lower_triangle_type);
    MPI_Type_commit(&lower_triangle_type);      // 提交数据类型

    if(rank == 0) {
        // 发送下三角部分到进程1
        MPI_Send(lower_triangle, 1, lower_triangle_type, 1, 0, MPI_COMM_WORLD);
        printf("发送者（rank 0）已发送下三角部分到接收者（rank 1）。\n");
    } else if(rank == 1) {
        // 接收下三角部分
        double* received_lower = (double*)malloc(count * sizeof(double));
        MPI_Recv(received_lower, 1, lower_triangle_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("接收者（rank 1）已接收下三角部分：\n");
        for(int i = 0; i < count; i++) {
            printf("%6.2f ", received_lower[i]);
            if((i+1) % (n) == 0) printf("\n");
        }
        printf("\n");
        free(received_lower);
    }

    // 释放资源
    MPI_Type_free(&lower_triangle_type);
    if(rank == 0) {
        free(A);
        free(lower_triangle);
    }

    MPI_Finalize();    // 结束MPI环境
    return 0;           // 程序结束
}