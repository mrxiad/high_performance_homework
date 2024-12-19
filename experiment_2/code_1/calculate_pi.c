#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;             // 定义进程编号和进程总数
    double a = 0.0, b = 1.0;    // 积分区间 [a, b]
    int n = 1000000;            // 定义总的积分子区间数
    double h = (b - a) / n;     // 计算每个小区间的宽度
    double local_sum = 0.0, total_sum = 0.0; // 定义本地和与全局和
    int i;

    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    // 获取当前进程的编号
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 获取进程总数
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 计算每个进程处理的子区间数
    int local_n = n / size;
    // 计算每个进程的起始和结束索引
    int start = rank * local_n;
    int end = start + local_n;

    // 进行本地积分计算
    for (i = start; i < end; i++) {
        double x = a + i * h + h / 2.0; // 计算当前子区间的中点
        local_sum += 4.0 / (1.0 + x * x); // 计算函数值并累加
    }

    // 使用MPI归约操作将所有本地和累加到总和
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // 只有主进程（rank 0）负责输出结果
    if (rank == 0) {
        double pi = h * total_sum; // 计算π的近似值
        printf("近似的π值为: %.16f\n", pi); // 输出结果
    }

    // 结束MPI环境
    MPI_Finalize();

    return 0; // 程序结束
}