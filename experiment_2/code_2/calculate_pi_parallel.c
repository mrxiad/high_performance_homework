// calculate_pi_parallel.c
#include <stdio.h>          // 标准输入输出库
#include <stdlib.h>         // 标准库
#include <omp.h>            // OpenMP库

int main() {
    double a = 0.0, b = 1.0;          // 积分区间 [a, b]
    int n = 100000000;                // 积分子区间数
    double h = (b - a) / n;           // 每个小区间的宽度
    double pi = 0.0;                   // 用于存储π的近似值

    // 开始并行区域
    #pragma omp parallel
    {
        double x, sum = 0.0; // 每个线程的局部和

        // 并行计算部分积分和
        #pragma omp for
        for (int i = 0; i < n; i++) {
            x = a + i * h + h / 2.0;           // 当前子区间的中点
            sum += 4.0 / (1.0 + x * x);       // 计算函数值并累加
        }

        // 使用原子操作累加到全局和
        #pragma omp atomic
        pi += sum;
    } // 结束并行区域

    pi *= h; // 计算最终的π值

    printf("近似的π值为: %.16f\n", pi); // 输出结果

    return 0; // 程序结束
}
