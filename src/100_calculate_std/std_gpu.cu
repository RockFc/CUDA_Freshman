#include "std_gpu.h"
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

#define THREADS_PER_BLOCK 256

// CUDA 计算均值 Kernel
__global__ void compute_mean(const float* data, float* mean, int N)
{
    __shared__ float sum[THREADS_PER_BLOCK];
    int              tid       = threadIdx.x + blockIdx.x * blockDim.x;
    int              local_tid = threadIdx.x;

    sum[local_tid] = (tid < N) ? data[tid] : 0.0f;
    __syncthreads();

    // 归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (local_tid < stride)
        {
            sum[local_tid] += sum[local_tid + stride];
        }
        __syncthreads();
    }

    if (local_tid == 0)
    {
        atomicAdd(mean, sum[0] / N);
    }
}

// CUDA 计算方差 Kernel
__global__ void compute_variance(const float* data, float mean, float* variance, int N)
{
    __shared__ float sum[THREADS_PER_BLOCK];
    int              tid       = threadIdx.x + blockIdx.x * blockDim.x;
    int              local_tid = threadIdx.x;

    sum[local_tid] = (tid < N) ? (data[tid] - mean) * (data[tid] - mean) : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (local_tid < stride)
        {
            sum[local_tid] += sum[local_tid + stride];
        }
        __syncthreads();
    }

    if (local_tid == 0)
    {
        atomicAdd(variance, sum[0] / N);
    }
}

// 静态方法实现
void CudaStats::ComputeMeanVariance(const std::vector<float>& data, float& mean, float& variance)
{
    int N = data.size();
    if (N == 0)
    {
        mean = variance = 0.0f;
        return;
    }

    // 分配 GPU 内存
    float *d_data, *d_mean, *d_variance;
    cudaMalloc(( void** )&d_data, N * sizeof(float));
    cudaMalloc(( void** )&d_mean, sizeof(float));
    cudaMalloc(( void** )&d_variance, sizeof(float));

    cudaMemcpy(d_data, data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_mean, 0, sizeof(float));
    cudaMemset(d_variance, 0, sizeof(float));

    // 计算块数
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // 运行均值计算
    compute_mean<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_data, d_mean, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);

    // 运行方差计算
    compute_variance<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_data, mean, d_variance, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&variance, d_variance, sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 资源
    cudaFree(d_data);
    cudaFree(d_mean);
    cudaFree(d_variance);
}