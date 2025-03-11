#include "hv/hthreadpool.h"
#include "hv/hv.h"
#include "to_string.h"
#include <gtest/gtest.h>
#include <thread>
#define private public
#define protected public
#include "std_gpu.h"
#undef private
#undef protected

size_t g_vecSize = 8192;
size_t g_deqSize = 5000;

class StdGpuTest : public testing::Test
{
public:
    void static SetUpTestCase()
    {
        HV_MEMCHECK;
        CreateData(g_v, false);
        CreateData(g_v0, true);
    }
    void static TearDownCase() {}

    static std::vector<std::deque<float>> g_v;
    static std::vector<std::deque<float>> g_v0;
    static std::vector<float>             g_avg;
    static std::vector<float>             g_std;
    std::shared_ptr<HThreadPool>          m_tp;
    std::mutex                            m_mtx;

protected:
    virtual void SetUp()
    {
        g_avg.clear();
        g_avg.reserve(g_vecSize);
        g_std.clear();
        g_std.reserve(g_vecSize);

        m_tp = std::make_shared<HThreadPool>();
        m_tp->start();
    }
    virtual void TearDown()
    {
        m_tp->stop();
        m_tp.reset();
    }
    static void CreateData(std::vector<std::deque<float>>& v, const bool& has0 = false)
    {
        for (size_t i = 1; i <= g_vecSize; i++)
        {
            std::deque<float> d;
            for (size_t j = 1; j <= g_deqSize; j++)
            {
                if (has0 && (j % 1000 == 0))
                {
                    d.push_back(0.0f);
                }
                else
                {
                    d.push_back(i * j);
                }
            }
            v.push_back(std::move(d));
        }
        // std::cout << helper::ToString(v) << std::endl;
    }
};
std::vector<std::deque<float>> StdGpuTest::g_v;
std::vector<std::deque<float>> StdGpuTest::g_v0;
std::vector<float>             StdGpuTest::g_avg;
std::vector<float>             StdGpuTest::g_std;

// 有小于等于0双入参标准差
TEST_F(StdGpuTest, test_CalcAvgAndStd_serial_1)
{
    common::gpu::CalcAvgAndStd(g_v0, true, g_avg, g_std);
    EXPECT_EQ(g_avg.size(), g_vecSize);
    EXPECT_EQ(g_std.size(), g_vecSize);
    std::cout << "g_avg: " << common::to_string(std::vector<float>(g_avg.begin(), g_avg.begin() + 10)) << std::endl;
    std::cout << "g_std: " << common::to_string(std::vector<float>(g_std.begin(), g_std.begin() + 10)) << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        g_vecSize = 8192;
        g_deqSize = 5000;
    }
    else
    {
        g_vecSize = atoi(argv[1]);
        g_deqSize = atoi(argv[2]);
    }
    printf("runing: %s g_vecSize[%d] g_deqSize[%d] \n", argv[0], g_vecSize, g_deqSize);

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
