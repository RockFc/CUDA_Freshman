#pragma once

#include <vector>

class CudaStats
{
public:
    // 计算均值和方差的静态方法
    static void ComputeMeanVariance(const std::vector<float>& data, float& mean, float& variance);
};