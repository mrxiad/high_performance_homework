# Linux(WSL) 系统并行计算环境配置教程

本教程旨在指导您在 Linux(WSL) 系统上配置并行计算环境，包括 MPI、OpenMP、pthreads 和 CUDA。完成配置后，您将能够在 Eclipse 或其他编辑器（如 VSCode、gedit）中编写和运行并行程序。

## 目录

1. [前置条件](#前置条件)
2. [安装必要的工具和依赖](#安装必要的工具和依赖)
3. [MPI 环境配置](#mpi-环境配置)
4. [OpenMP 环境配置](#openmp-环境配置)
5. [pthreads 环境配置](#pthreads-环境配置)
6. [CUDA 环境配置](#cuda-环境配置)
7. [配置 Eclipse](#配置-eclipse)
8. [验证安装](#验证安装)

------

## 前置条件

- **操作系统**：基于 Debian 的 Linux 发行版（如 Ubuntu 20.04 及以上）
- **用户权限**：具有 sudo 权限
- **网络连接**：确保系统能够访问互联网以下载必要的软件包

## 安装必要的工具和依赖

在开始之前，确保您的系统是最新的，并安装了一些基本工具。

```
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential git curl wget
```

## MPI 环境配置

MPI（Message Passing Interface）用于在分布式系统中进行进程间通信。

### 安装 OpenMPI

```
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
```

### 配置环境变量

通常，安装 OpenMPI 后，环境变量会自动配置。如果需要手动配置，可以添加以下内容到 `~/.bashrc` 或 `~/.zshrc`：

```
export PATH=/usr/lib/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/openmpi/lib:$LD_LIBRARY_PATH
```

然后，刷新终端：

```
source ~/.bashrc
```

### 验证安装

编写一个简单的 MPI 程序 `mpi_hello.c`：

```
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv){
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Hello world from processor %s, rank %d out of %d processors\n",processor_name, world_rank, world_size);

    MPI_Finalize();
}
```

编译并运行：

```
mpicc mpi_hello.c -o mpi_hello
mpirun -np 4 ./mpi_hello
```



## OpenMP 环境配置

OpenMP 是一种用于共享内存多线程编程的 API。

### 安装 GCC 支持 OpenMP

GCC 通常已经支持 OpenMP。确保安装了最新版本的 GCC：

```
sudo apt install -y gcc g++
```

### 编写和编译 OpenMP 程序

示例程序 `openmp_hello.c`：

```
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    int nthreads, tid;

    /* Fork a team of threads giving them their own copies of variables */
    #pragma omp parallel private(nthreads, tid)
    {

        /* Obtain thread number */
        tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);

        /* Only master thread does this */
        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }

    }  /* All threads join master thread and disband */
    return 0;
}
```

编译并运行：

```
gcc -fopenmp openmp_hello.c -o openmp_hello
./openmp_hello
```

### 设置线程数

可以通过环境变量 `OMP_NUM_THREADS` 来设置线程数量：

```
export OMP_NUM_THREADS=4
./openmp_hello
```

## pthreads 环境配置

pthreads 是 POSIX 标准的多线程编程库。

### 编写和编译 pthreads 程序

示例程序 `pthread_hello.c`：

```
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
 
#define THREAD_NUMBER 2
 
int retval_hello1 = 1;
int retval_hello2 = 2;
 
void* hello1(void* arg){
    char* hello_str = (char *)arg;
    sleep(1);
    printf("%s\n", hello_str);
    pthread_exit(&retval_hello1);
}
 
void* hello2(void* arg){
    char* hello_str = (char *)arg;
    sleep(2);
    printf("%s\n", hello_str);
    pthread_exit(&retval_hello2);
}
 
int main()
{
    int retval;
    int *retval_hello[2];
 
    pthread_t pt[2];
    const char* arg[THREAD_NUMBER];
    arg[0] = "hello world from thread1.";
    arg[1] = "hello world from thread2.";
    printf("begin to create threads....\n");
 
    retval = pthread_create(&pt[0], NULL, hello1, (void*)arg[0]);
    if(retval != 0){
        printf("pthread_create error.");
        exit(1);
    }
    retval = pthread_create(&pt[1], NULL, hello2, (void*)arg[1]);
    if(retval != 0){
        printf("pthread_create error.");
        exit(1);
    }
 
    printf("now, the main thread returns.\n");
    printf("main thread begins to wait threads.\n");
    for(int i=0;i<THREAD_NUMBER;i++){
        retval = pthread_join(pt[i], (void **)&retval_hello[i]);
        if(retval != 0){
            printf("pthread_join error");
            exit(1);
        }else{
            printf("return value is %d\n", *retval_hello[i]);
        }
    }
    return 0;
}
```

编译并运行：

```
gcc -pthread pthread_hello.c -o pthread_hello
./pthread_hello
```





## CUDA 环境配置

[csdn教程]: https://blog.csdn.net/weixin_51942493/article/details/127544188?ops_request_misc=%257B%2522request%255Fid%2522%253A%25221972c253a0b25091ba3b6e6dd62d9c21%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&amp;request_id=1972c253a0b25091ba3b6e6dd62d9c21&amp;biz_id=0&amp;utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-127544188-null-null.142^v100^control&amp;utm_term=wsl%E5%AE%89%E8%A3%85cuda&amp;spm=1018.2226.3001.4187

CUDA 用于 NVIDIA GPU 的并行计算。

### 检查 NVIDIA GPU 和驱动

确保wsl支持NVIDIA GPU 

```bash
nvidia-smi

# 输出    
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.35                 Driver Version: 546.30       CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060 ...    On  | 00000000:01:00.0  On |                  N/A |
| N/A   45C    P8              16W / 130W |   1364MiB /  6144MiB |     12%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A        35      G   /Xwayland                                 N/A      |
+---------------------------------------------------------------------------------------+
```



### 安装 CUDA Toolkit

1. **下载 CUDA Toolkit**：

   访问 [NVIDIA CUDA Toolkit 下载页面](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

2. **安装 CUDA Toolkit**：

   执行以下命令：

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
   sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
   sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-6
   ```

   在安装过程中，您可以选择安装驱动（如果尚未安装）和 CUDA 工具包。、

3. **配置环境变量**：

   添加以下内容到 `~/.bashrc` 或 `~/.zshrc`：

   ```
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

   然后，刷新终端：

   ```
   source ~/.bashrc
   ```

### 验证安装

编写一个简单的 CUDA 程序 `cuda_hello.cu`：

```
#include<stdio.h>
__global__ void cuda_hello()
{
    printf("Hello World from GPU!\n");
}

int main() 
{
    cuda_hello<<<2,2>>>(); 
    cudaDeviceReset();
    return 0;
}
```

编译并运行：

```
nvcc cuda_hello.cu -o cuda_hello
./cuda_hello
```