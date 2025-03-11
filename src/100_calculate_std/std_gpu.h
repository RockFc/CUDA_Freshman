#pragma once
#include <vector>

namespace common::gpu
{
    template <typename NestedContainer>
    void CalcAvgAndStd(const NestedContainer& data,
                       bool                   ignoreNonPositive,
                       std::vector<float>&    outputAvg,
                       std::vector<float>&    outputStd);
}  // namespace common::gpu