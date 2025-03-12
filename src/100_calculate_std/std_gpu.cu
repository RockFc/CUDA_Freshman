#include "std_gpu.h"
#include <cmath>
#include <cuda_runtime.h>
#include <deque>
#include <iostream>
#include <vector>

#define THREADS_PER_BLOCK 256

namespace common::gpu
{

// CUDA 核函数：批量计算均值
__global__ void compute_mean_batch(const float* data, int* offsets, int* sizes, float* means, int N)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int start = offsets[row];
    int size  = sizes[row];

    __shared__ float sum[THREADS_PER_BLOCK];

    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        local_sum += data[start + i];
    }

    sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            sum[tid] += sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        means[row] = sum[0] / size;
    }
}

// CUDA 核函数：批量计算标准差
__global__ void compute_std_batch(
    const float* data, int* offsets, int* sizes, const float* means, float* stddevs, int N)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int start = offsets[row];
    int size  = sizes[row];

    __shared__ float sum[THREADS_PER_BLOCK];

    float mean      = means[row];
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        float diff = data[start + i] - mean;
        local_sum += diff * diff;
    }

    sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            sum[tid] += sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        stddevs[row] = sqrtf(sum[0] / size);
    }
}

template <typename NestedContainer>
static void CalcAvgAndStd(const NestedContainer& data,
                          bool                   ignoreNonPositive,
                          std::vector<float>&    outputAvg,
                          std::vector<float>&    outputStd)
{
    outputAvg.clear();
    outputStd.clear();

    std::vector<int> sizes;
    int              totalFilteredSize = 0;

    // 预先计算总大小
    for (const auto& vec : data)
    {
        int filteredSize = 0;
        for (float val : vec)
        {
            if (!ignoreNonPositive || val > 0)
            {
                filteredSize++;
            }
        }
        sizes.push_back(filteredSize);
        totalFilteredSize += filteredSize;
    }

    if (totalFilteredSize == 0)
    {
        outputAvg.resize(data.size(), 0.0f);
        outputStd.resize(data.size(), 0.0f);
        return;
    }

    std::vector<float> flatData(totalFilteredSize);
    std::vector<int>   offsets(data.size());

    int currentOffset = 0;
    for (size_t i = 0; i < data.size(); ++i)
    {
        offsets[i] = currentOffset;
        int index  = 0;
        for (float val : data[i])
        {
            if (!ignoreNonPositive || val > 0)
            {
                flatData[currentOffset + index] = val;
                index++;
            }
        }
        currentOffset += sizes[i];
    }

    int N = data.size();

    // 申请 GPU 内存
    float *d_data, *d_means, *d_stddevs;
    int *  d_offsets, *d_sizes;
    cudaMalloc(&d_data, totalFilteredSize * sizeof(float));
    cudaMalloc(&d_means, N * sizeof(float));
    cudaMalloc(&d_stddevs, N * sizeof(float));
    cudaMalloc(&d_offsets, N * sizeof(int));
    cudaMalloc(&d_sizes, N * sizeof(int));

    cudaMemcpy(d_data, flatData.data(), totalFilteredSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, sizes.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // 计算均值
    compute_mean_batch<<<N, THREADS_PER_BLOCK>>>(d_data, d_offsets, d_sizes, d_means, N);
    cudaDeviceSynchronize();

    // 计算标准差
    compute_std_batch<<<N, THREADS_PER_BLOCK>>>(d_data, d_offsets, d_sizes, d_means, d_stddevs, N);
    cudaDeviceSynchronize();

    // 取回数据
    outputAvg.resize(N);
    outputStd.resize(N);
    cudaMemcpy(outputAvg.data(), d_means, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputStd.data(), d_stddevs, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 资源
    cudaFree(d_data);
    cudaFree(d_means);
    cudaFree(d_stddevs);
    cudaFree(d_offsets);
    cudaFree(d_sizes);
}

// 显式实例化模板
template void CalcAvgAndStd<std::vector<std::deque<float>>>(const std::vector<std::deque<float>>&,
                                                            bool,
                                                            std::vector<float>&,
                                                            std::vector<float>&);

}  // namespace common::gpu