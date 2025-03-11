#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <execution>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace common::cpu
{

/*
 * 入参为 Container 的计算函数中，Container 支持 std::vector, std::deque,
 * std::list 等容器 入参为 NestedContainer 的计算函数中，NestedContainer 支持
 * std::vector<std::deque<float>>, std::vector<std::vector<float>>,
 * std::list<std::vector<float>> 等两层容器嵌套
 */

// 过滤小于等于0的值，并返回容器的大小【不建议使用】
template <typename Container>
static std::size_t FilterNonPositive(Container& data)
{
    if (data.empty())
    {
        return 0;
    }

    static_assert(std::is_floating_point<typename Container::value_type>::value,
                  "Container's value_type must be a floating-point type.");

    auto newEnd
        = std::remove_if(data.begin(), data.end(), [](const typename Container::value_type& i) {
              return (i <= 0.0f);
          });
    data.erase(newEnd, data.end());
    return data.size();
}

// 计算平均值【不建议使用】
template <typename Container>
static float CalcAvg(const Container& data)
{
    if (data.empty())
    {
        return 0.0f;
    }

    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return static_cast<float>(sum / data.size());
}

// 计算标准差【不建议使用】
template <typename Container>
static float CalcStd(const Container& data)
{
    if (data.size() <= 1)
    {
        return 0.0f;
    }

    double mean = CalcAvg(data);
    double variance
        = std::accumulate(data.begin(), data.end(), 0.0, [mean](double accum, float value) {
              return accum + (value - mean) * (value - mean);
          });

    return static_cast<float>(std::sqrt(variance / data.size()));
}

// 计算总和与元素个数，支持是否忽略小于等于0的值
template <typename Container>
static std::tuple<double, std::size_t> CalcSumAndCount(const Container& data,
                                                       bool             ignoreNonPositive)
{
    double      sum   = 0.0;
    std::size_t count = 0;
    for (const auto& value : data)
    {
        if (!ignoreNonPositive || value > 0.0f)
        {
            sum += value;
            ++count;
        }
    }
    return {sum, count};
}

// 计算方差，支持是否忽略小于等于0的值
template <typename Container>
static double CalcVariance(const Container& data, double mean, bool ignoreNonPositive)
{
    if (data.size() <= 1)
    {
        return 0.0;
    }

    double variance = 0.0;
    for (const auto& value : data)
    {
        if (!ignoreNonPositive || value > 0.0f)
        {
            double diff = value - mean;
            variance += diff * diff;
        }
    }

    return variance;
}

// 计算平均值，支持是否忽略小于等于0的值
template <typename Container>
static float CalcAvg(const Container& data, bool ignoreNonPositive)
{
    if (data.empty())
    {
        return 0.0f;
    }

    auto [sum, count] = CalcSumAndCount(data, ignoreNonPositive);
    return count == 0 ? 0.0f : static_cast<float>(sum / count);
}

// 计算均值和标准差，支持是否忽略小于等于0的值
template <typename Container>
static void CalcAvgAndStd(const Container& data, bool ignoreNonPositive, float& avg, float& stdDev)
{
    if (data.size() <= 1)
    {
        avg    = 0.0f;
        stdDev = 0.0f;
        return;
    }

    // 求和
    auto [sum, count] = CalcSumAndCount(data, ignoreNonPositive);
    if (count == 0)
    {
        avg    = 0.0f;
        stdDev = 0.0f;
        return;
    }
    avg = static_cast<float>(sum / count);

    // 计算方差
    double variance = CalcVariance(data, avg, ignoreNonPositive);

    // 计算标准差
    stdDev = static_cast<float>(std::sqrt(variance / count));
}

// 计算标准差，支持是否忽略小于等于0的值
template <typename Container>
static float CalcStd(const Container& data, bool ignoreNonPositive)
{
    float avg, stdDev;
    CalcAvgAndStd(data, ignoreNonPositive, avg, stdDev);
    return stdDev;
}

// 计算平均值，并将结果保存到 outputAvg
template <typename NestedContainer>
static void
CalcAvg(const NestedContainer& data, bool ignoreNonPositive, std::vector<float>& outputAvg)
{
    outputAvg.clear();
    outputAvg.reserve(data.size());
    std::for_each(data.begin(), data.end(), [&](const auto& innerContainer) {
        outputAvg.emplace_back(CalcAvg(innerContainer, ignoreNonPositive));
    });
}

// 计算标准差，并将结果保存到 outputStd
template <typename NestedContainer>
static void
CalcStd(const NestedContainer& data, bool ignoreNonPositive, std::vector<float>& outputStd)
{
    outputStd.clear();
    outputStd.reserve(data.size());
    std::for_each(data.begin(), data.end(), [&](const auto& innerContainer) {
        outputStd.emplace_back(CalcStd(innerContainer, ignoreNonPositive));
    });
}

// 计算均值和标准差，并分别将结果保存到 outputAvg 和 outputStd
template <typename NestedContainer>
static void CalcAvgAndStd(const NestedContainer& data,
                          bool                   ignoreNonPositive,
                          std::vector<float>&    outputAvg,
                          std::vector<float>&    outputStd)
{
    outputAvg.clear();
    outputStd.clear();
    outputAvg.reserve(data.size());
    outputStd.reserve(data.size());

    std::for_each(data.begin(), data.end(), [&](const auto& innerContainer) {
        float avgResult, stdResult;
        CalcAvgAndStd(innerContainer, ignoreNonPositive, avgResult, stdResult);
        outputAvg.emplace_back(avgResult);
        outputStd.emplace_back(stdResult);
    });
}

// 计算平均值，并将结果保存到 outputAvg
template <typename NestedContainer>
static void
CalcAvgParallel(const NestedContainer& data, bool ignoreNonPositive, std::vector<float>& outputAvg)
{
    outputAvg.clear();
    outputAvg.reserve(data.size());  // 预分配空间

    // 并行计算每个内层容器的平均值
    std::for_each(std::execution::par, data.begin(), data.end(),
                  [&outputAvg, ignoreNonPositive](const auto& innerContainer) {
                      outputAvg.emplace_back(CalcAvg(innerContainer, ignoreNonPositive));
                  });
}

// 计算标准差，并将结果保存到 outputStd
template <typename NestedContainer>
static void
CalcStdParallel(const NestedContainer& data, bool ignoreNonPositive, std::vector<float>& outputStd)
{
    outputStd.clear();
    outputStd.reserve(data.size());  // 预分配空间

    // 并行计算每个内层容器的标准差
    std::for_each(std::execution::par, data.begin(), data.end(),
                  [&outputStd, ignoreNonPositive](const auto& innerContainer) {
                      outputStd.emplace_back(CalcStd(innerContainer, ignoreNonPositive));
                  });
}

// 计算均值和标准差，并分别将结果保存到 outputAvg 和 outputStd
template <typename NestedContainer>
static void CalcAvgAndStdParallel(const NestedContainer& data,
                                  bool                   ignoreNonPositive,
                                  std::vector<float>&    outputAvg,
                                  std::vector<float>&    outputStd)
{
    outputAvg.clear();
    outputStd.clear();
    outputAvg.reserve(data.size());
    outputStd.reserve(data.size());

    // 并行计算每个内层容器的均值和标准差
    std::for_each(std::execution::par, data.begin(), data.end(),
                  [&outputAvg, &outputStd, ignoreNonPositive](const auto& innerContainer) {
                      float avgResult, stdResult;
                      CalcAvgAndStd(innerContainer, ignoreNonPositive, avgResult, stdResult);
                      outputAvg.emplace_back(avgResult);
                      outputStd.emplace_back(stdResult);
                  });
}

}  // namespace common::cpu