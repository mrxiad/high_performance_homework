# 提示词

（1） 编写通用矩阵乘法（GEMM）的并行程序。（30分）

输入：M、N、K三个整数（矩阵规模128 ~1024）

问题描述：随机生成大小为MN和NK的两个矩阵A、B，对这两个矩阵做乘法得到矩阵C

输出：A、B、C 三个矩阵以及矩阵计算的时间

1）用MPI实现通用矩阵乘法的高效并行计算（10分）

2）用MPI+OpenMP实现通用矩阵乘法的高效并行计算（10分）

3）用CUDA实现通用矩阵乘法的高效异构并行计算（10分）

你的回复需要符合包含以下内容:

1.创建文件的指令"touch ..."

2.每个程序运用的公式

3.每个程序的代码(c风格+注释)

4.编译指令和运行指令,以及对这些的解释



# 回复

好的，以下是针对通用矩阵乘法（GEMM）的三种高效并行实现方案，包括：

1. **使用MPI实现通用矩阵乘法的高效并行计算**
2. **使用MPI+OpenMP实现通用矩阵乘法的高效并行计算**
3. **使用CUDA实现通用矩阵乘法的高效异构并行计算**

每个实现都包含创建文件的指令、所用的数学公式、C/CUDA代码（带详细注释）、编译和运行指令及其解释。

------

## 1. 使用MPI实现通用矩阵乘法的高效并行计算

### 1. 创建文件的指令

使用`touch`命令创建MPI矩阵乘法的源文件：

```bash
touch mpi_gemm.c
```

### 2. 使用的数学公式

矩阵乘法

### 3. C语言代码（带详细注释）

以下是`mpi_gemm.c`的完整代码，实现了使用MPI进行通用矩阵乘法的高效并行计算：

```c
// mpi_gemm.c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
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

    // 执行本地的矩阵乘法
    for(int i = 0; i < rows_per_proc; i++){
        for(int j = 0; j < K; j++){
            local_C[i*K + j] = 0.0;
            for(int l = 0; l < N; l++){
                local_C[i*K + j] += local_A[i*N + l] * B[l*K + j];
            }
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
        printf("Matrix multiplication completed in %f seconds.\n", end_time - start_time);
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
```

### 4. 编译指令和运行指令及解释

#### 编译指令

使用`mpicc`编译器编译MPI矩阵乘法程序：

```bash
mpicc -o mpi_gemm mpi_gemm.c
```

**解释：**

- `mpicc`：MPI的C语言编译器包装器，负责链接MPI库。
- `-o mpi_gemm`：指定输出的可执行文件名为`mpi_gemm`。
- `mpi_gemm.c`：源代码文件。

#### 运行指令

使用`mpirun`或`mpiexec`运行编译后的MPI程序。假设矩阵维度为M=512, N=512, K=512，使用4个进程：

```bash
mpirun -np 4 ./mpi_gemm 512 512 512
```

**解释：**

- `mpirun`或`mpiexec`：用于启动MPI程序的命令。
- `-np 4`：指定使用4个MPI进程。
- `./mpi_gemm`：执行编译后的程序。
- `512 512 512`：命令行参数，分别代表矩阵A的维度M×N和矩阵B的维度N×K。

**示例输出：**

```bash
Matrix multiplication completed in 2.345678 seconds.
```

**注意事项：**

- **矩阵大小**：矩阵维度M, N, K应在128到1024之间，可以根据需要调整。
- **进程数**：确保M能够被进程数整除，以避免数据分配不均。
- **打印矩阵**：代码中有打印矩阵A、B和C的函数调用被注释掉，适用于小规模矩阵调试。对于大规模矩阵，建议保留注释以避免大量输出。

------

## 2. 使用MPI+OpenMP实现通用矩阵乘法的高效并行计算

### 1. 创建文件的指令

使用`touch`命令创建MPI+OpenMP矩阵乘法的源文件：

```bash
touch mpi_openmp_gemm.c
```

### 2. 使用的数学公式

同上

### 3. C语言代码（带详细注释）

以下是`mpi_openmp_gemm.c`的完整代码，实现了使用MPI和OpenMP进行通用矩阵乘法的高效并行计算：

```c
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
```

### 4. 编译指令和运行指令及解释

#### 编译指令

使用支持MPI和OpenMP的`mpicc`编译器编译MPI+OpenMP矩阵乘法程序：

```bash
mpicc -fopenmp -o mpi_openmp_gemm mpi_openmp_gemm.c
```

**解释：**

- `mpicc`：MPI的C语言编译器包装器，负责链接MPI库。
- `-fopenmp`：启用OpenMP支持，允许编译器识别和处理OpenMP指令。
- `-o mpi_openmp_gemm`：指定输出的可执行文件名为`mpi_openmp_gemm`。
- `mpi_openmp_gemm.c`：源代码文件。

#### 运行指令

使用`mpirun`或`mpiexec`运行编译后的MPI+OpenMP程序。假设矩阵维度为M=512, N=512, K=512，使用4个MPI进程，每个进程使用4个OpenMP线程：

```bash
export OMP_NUM_THREADS=4
mpirun -np 4 ./mpi_openmp_gemm 512 512 512
```

**解释：**

- `export OMP_NUM_THREADS=4`：设置OpenMP程序每个MPI进程使用4个线程。
- `mpirun`或`mpiexec`：用于启动MPI程序的命令。
- `-np 4`：指定使用4个MPI进程。
- `./mpi_openmp_gemm`：执行编译后的程序。
- `512 512 512`：命令行参数，分别代表矩阵A的维度M×N和矩阵B的维度N×K。

**示例输出：**

```bash
MPI+OpenMP Matrix multiplication completed in 1.234567 seconds.
```

**注意事项：**

- **矩阵大小**：矩阵维度M, N, K应在128到1024之间，可以根据需要调整。
- **进程数和线程数**：确保M能够被进程数整除，并根据硬件资源合理设置MPI进程数和OpenMP线程数，以最大化并行效率。
- **打印矩阵**：代码中有打印矩阵A、B和C的函数调用被注释掉，适用于小规模矩阵调试。对于大规模矩阵，建议保留注释以避免大量输出。

------

## 3. 使用CUDA实现通用矩阵乘法的高效异构并行计算

### 1. 创建文件的指令

使用`touch`命令创建CUDA矩阵乘法的源文件：

```bash
touch cuda_gemm.cu
```

### 2. 使用的数学公式

同上

### 3. CUDA C代码（带详细注释）

以下是`cuda_gemm.cu`的完整代码，实现了使用CUDA进行通用矩阵乘法的高效异构并行计算：

```c
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
```

### 4. 编译指令和运行指令及解释

#### 编译指令

使用`nvcc`编译器编译CUDA矩阵乘法程序：

```bash
nvcc -o cuda_gemm cuda_gemm.cu
```

> 上述指令可能正常编译，但是运行无结果，需要改为`nvcc -o cuda_gemm cuda_gemm.cu -arch=sm_86`

**解释：**

- `nvcc`：NVIDIA的CUDA C编译器。
- `-o cuda_gemm`：指定输出的可执行文件名为`cuda_gemm`。
- `cuda_gemm.cu`：源代码文件。

#### 运行指令

运行编译后的CUDA矩阵乘法程序。假设矩阵维度为M=512, N=512, K=512：

```bash
./cuda_gemm 512 512 512
```

**解释：**

- `./cuda_gemm`：执行编译后的程序。
- `512 512 512`：命令行参数，分别代表矩阵A的维度M×N和矩阵B的维度N×K。

**示例输出：**

```bash
CUDA Matrix multiplication completed in 0.123456 seconds.
```

**注意事项：**

- **CUDA设备**：确保系统中安装了支持CUDA的NVIDIA GPU，并已正确安装CUDA Toolkit。
- **矩阵大小**：矩阵维度M, N, K应在128到1024之间，可以根据需要调整。
- **打印矩阵**：代码中有打印矩阵A、B和C的部分被注释掉，适用于小规模矩阵调试。对于大规模矩阵，建议保留注释以避免大量输出。
- **时间测量**：使用`clock()`函数测量主机代码执行时间，并不包括设备内核执行时间的精确测量。对于更精确的CUDA性能分析，建议使用CUDA事件（`cudaEvent_t`）进行时间测量。

------

## 总结

以上三种实现方案分别利用了MPI、MPI+OpenMP和CUDA三种并行编程模型，针对通用矩阵乘法（GEMM）提供了高效的并行计算方法。具体特点如下：

1. **MPI实现**：
   - **适用场景**：适用于分布式内存系统，通过消息传递进行数据通信。
   - **优势**：可扩展性强，适用于大规模并行计算。
   - **劣势**：需要显式管理数据分布和通信。
2. **MPI+OpenMP实现**：
   - **适用场景**：结合了MPI的分布式内存并行和OpenMP的共享内存并行，适用于多节点多核心系统。
   - **优势**：在分布式系统中充分利用多核处理器，提高计算效率。
   - **劣势**：编程复杂度增加，需要同时管理MPI和OpenMP的并行结构。
3. **CUDA实现**：
   - **适用场景**：适用于具有NVIDIA GPU的系统，通过GPU进行大规模并行计算。
   - **优势**：利用GPU的高并行度，实现极高的计算性能。
   - **劣势**：需要具备CUDA编程知识，且依赖于特定硬件（NVIDIA GPU）。





### 性能和效率考虑

- **数据分布**：在MPI和MPI+OpenMP实现中，合理的数据分布和负载均衡对性能至关重要。
- **并行粒度**：选择合适的并行粒度（如块级、行级）能够有效提升并行效率。
- **内存访问模式**：优化内存访问模式，减少缓存未命中和内存带宽瓶颈。
- CUDA优化:
  - **内存管理**：利用共享内存和内存访问优化，减少全局内存访问延迟。
  - **线程块配置**：选择合适的线程块大小，提高GPU资源利用率。
  - **并行度**：充分利用GPU的并行计算能力，通过优化核函数实现高效计算。

通过选择合适的并行编程模型和优化策略，可以显著提升通用矩阵乘法的计算性能，满足不同计算平台和应用需求。

如果有更多问题或需要进一步的帮助，欢迎继续提问！