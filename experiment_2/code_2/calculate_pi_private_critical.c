// calculate_pi_private_critical.c
#include <stdio.h>          // 标准输入输出库
#include <stdlib.h>         // 标准库
#include <omp.h>            // OpenMP库

int main() {
    double a = 0.0, b = 1.0;          // 积分区间 [a, b]
    int n = 100000000;                // 积分子区间数
    double h = (b - a) / n;           // 每个小区间的宽度
    double pi = 0.0;                   // 用于存储π的近似值

    // 并行for循环，使用private和critical进行同步
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double x = a + i * h + h / 2.0;  // 当前子区间的中点
        double f = 4.0 / (1.0 + x * x);  // 函数值

        // 临界区，确保对全局变量pi的安全更新
        #pragma omp critical
        {
            pi += f;
        }
    }

    pi *= h; // 计算最终的π值

    printf("近似的π值为: %.16f\n", pi); // 输出结果

    return 0; // 程序结束
}