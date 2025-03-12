#include "hv/hthreadpool.h"
#include "hv/hv.h"
#include "to_string.h"
#include <gtest/gtest.h>
#include <thread>
#define private public
#define protected public
#include "std_cpu.h"
#undef private
#undef protected

size_t g_vecSize = 8192;
size_t g_deqSize = 5000;

class StdCpuTest : public testing::Test
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
std::vector<std::deque<float>> StdCpuTest::g_v;
std::vector<std::deque<float>> StdCpuTest::g_v0;
std::vector<float>             StdCpuTest::g_avg;
std::vector<float>             StdCpuTest::g_std;

// 无小于等于0单入参均值
TEST_F(StdCpuTest, test_0)
{
    const std::deque<float> d{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::cout << "CalcAvg(d):" << common::cpu::CalcAvg(d) << std::endl;
    std::cout << "CalcAvg(d, false):" << common::cpu::CalcAvg(d, false) << std::endl;
    std::cout << "CalcStd(d):" << common::cpu::CalcStd(d) << std::endl;
    std::cout << "CalcStd(d, false):" << common::cpu::CalcStd(d, false) << std::endl;

    std::deque<float> d0{0.0f, 0.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    common::cpu::FilterNonPositive(d0);
    std::cout << "CalcAvg(d0):" << common::cpu::CalcAvg(d0) << std::endl;
    d0 = {0.0f, 0.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::cout << "CalcAvg(d0, true):" << common::cpu::CalcAvg(d0, true) << std::endl;
    d0 = {0.0f, 0.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    common::cpu::FilterNonPositive(d0);
    std::cout << "CalcStd(d0):" << common::cpu::CalcStd(d0) << std::endl;
    d0 = {0.0f, 0.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::cout << "CalcStd(d0, true):" << common::cpu::CalcStd(d0, true) << std::endl;

    /*
  CalcAvg(d):5
  CalcAvg(d, false):5
  CalcStd(d):2.58199
  CalcStd(d, false):2.58199
  CalcAvg(d0):6
  CalcAvg(d0, true):6
  CalcStd(d0):2
  CalcStd(d0, true):2
    */
}

// 无小于等于0单入参均值
TEST_F(StdCpuTest, test_CalcAvg1_1)
{
    for (auto it = g_v.begin(); it != g_v.end(); it++)
    {
        g_avg.push_back(std::move(common::cpu::CalcAvg(*it)));
    }
    EXPECT_EQ(g_avg.size(), g_vecSize);
}

// 无小于等于0双入参均值
TEST_F(StdCpuTest, test_CalcAvg2_1)
{
    for (auto it = g_v.begin(); it != g_v.end(); it++)
    {
        g_avg.push_back(std::move(common::cpu::CalcAvg(*it, false)));
    }
    EXPECT_EQ(g_avg.size(), g_vecSize);
}

// 无小于等于0单入参标准差
TEST_F(StdCpuTest, test_CalcStd1_1)
{
    for (auto it = g_v.begin(); it != g_v.end(); it++)
    {
        g_std.push_back(std::move(common::cpu::CalcStd(*it)));
    }
    EXPECT_EQ(g_std.size(), g_vecSize);
}

// 无小于等于0双入参标准差
TEST_F(StdCpuTest, test_CalcStd2_1)
{
    for (auto it = g_v.begin(); it != g_v.end(); it++)
    {
        g_std.push_back(std::move(common::cpu::CalcStd(*it, false)));
    }
    EXPECT_EQ(g_std.size(), g_vecSize);
}

/*
[ RUN      ] StdCpuTest.test_CalcAvg1_1
[       OK ] StdCpuTest.test_CalcAvg1_1 (139 ms)
[ RUN      ] StdCpuTest.test_CalcAvg2_1
[       OK ] StdCpuTest.test_CalcAvg2_1 (162 ms)
[ RUN      ] StdCpuTest.test_CalcStd1_1
[       OK ] StdCpuTest.test_CalcStd1_1 (311 ms)
[ RUN      ] StdCpuTest.test_CalcStd2_1
[       OK ] StdCpuTest.test_CalcStd2_1 (312 ms)
小结：
1.如果数据中无小于等于0的值，单入参速度较快，符合预期，因为不需要判断标志位
*/

// 有小于等于0单入参均值
TEST_F(StdCpuTest, test_CalcAvg1_2)
{
    for (auto it = g_v0.begin(); it != g_v0.end(); it++)
    {
        common::cpu::FilterNonPositive(*it);
        g_avg.push_back(std::move(common::cpu::CalcAvg(*it)));
    }
    EXPECT_EQ(g_avg.size(), g_vecSize);
}

// 有小于等于0双入参均值
TEST_F(StdCpuTest, test_CalcAvg2_2)
{
    for (auto it = g_v0.begin(); it != g_v0.end(); it++)
    {
        g_avg.push_back(std::move(common::cpu::CalcAvg(*it, true)));
    }
    EXPECT_EQ(g_avg.size(), g_vecSize);
}

// 有小于等于0单入参标准差
TEST_F(StdCpuTest, test_CalcStd1_2)
{
    for (auto it = g_v0.begin(); it != g_v0.end(); it++)
    {
        common::cpu::FilterNonPositive(*it);
        g_std.push_back(std::move(common::cpu::CalcStd(*it)));
    }
    EXPECT_EQ(g_std.size(), g_vecSize);
}

// 有小于等于0双入参标准差
TEST_F(StdCpuTest, test_CalcStd2_2)
{
    for (auto it = g_v0.begin(); it != g_v0.end(); it++)
    {
        g_std.push_back(std::move(common::cpu::CalcStd(*it, true)));
    }
    EXPECT_EQ(g_std.size(), g_vecSize);
}

// 有小于等于0双入参同时计算均值和标准差
TEST_F(StdCpuTest, test_CalcAvgAndStd_2)
{
    float avg, std;
    for (auto it = g_v0.begin(); it != g_v0.end(); it++)
    {
        common::cpu::CalcAvgAndStd(*it, true, avg, std);
        g_avg.push_back(avg);
        g_std.push_back(std);
    }
    EXPECT_EQ(g_avg.size(), g_vecSize);
    EXPECT_EQ(g_std.size(), g_vecSize);
}

/*
[ RUN      ] StdCpuTest.test_CalcAvg1_2
[       OK ] StdCpuTest.test_CalcAvg1_2 (467 ms)
[ RUN      ] StdCpuTest.test_CalcAvg2_2
[       OK ] StdCpuTest.test_CalcAvg2_2 (175 ms)
[ RUN      ] StdCpuTest.test_CalcStd1_2
[       OK ] StdCpuTest.test_CalcStd1_2 (491 ms)
[ RUN      ] StdCpuTest.test_CalcStd2_2
[       OK ] StdCpuTest.test_CalcStd2_2 (336 ms)
[ RUN      ] StdCpuTest.test_CalcAvgAndStd_2
[       OK ] StdCpuTest.test_CalcAvgAndStd_2 (335 ms)
小结：
1.如果数据中无小于等于0的值，单入参速度较快，符合预期，因为不需要判断标志位
2.如果数据中有小于等于0的值，单入参明显较慢（例如test_CalcAvg1_2计算均值用例），不建议使用，因为需要先删除小于等于0的元素
3.业务中如果需要同时计算均值和方差，直接可使用 CalcAvgAndStd
版本，节省单独计算均值的时间 4.生产环境中优先选择双入参的版本
*/

// 有小于等于0双入参均值
TEST_F(StdCpuTest, test_CalcAvg2_thredpool_1)
{
    for (auto it = g_v0.begin(); it != g_v0.end(); it++)
    {
        m_tp->commit([it, this]() {
            std::lock_guard<std::mutex> lock(m_mtx);
            g_avg.push_back(std::move(common::cpu::CalcAvg(*it, true)));
        });
    }
    m_tp->wait();
    EXPECT_EQ(g_avg.size(), g_vecSize);
}

// 有小于等于0双入参标准差
TEST_F(StdCpuTest, test_CalcStd2_thredpool_1)
{
    for (auto it = g_v0.begin(); it != g_v0.end(); it++)
    {
        m_tp->commit([it, this]() {
            std::lock_guard<std::mutex> lock(m_mtx);
            g_std.push_back(std::move(common::cpu::CalcStd(*it, true)));
        });
    }
    m_tp->wait();
    EXPECT_EQ(g_std.size(), g_vecSize);
}

// 有小于等于0双入参均值
TEST_F(StdCpuTest, test_CalcAvg2_thredpool_2)
{
    m_tp->commit([]() {
        for (auto it = g_v0.begin(); it != g_v0.end(); it++)
        {
            g_avg.push_back(std::move(common::cpu::CalcAvg(*it, true)));
        }
    });
    m_tp->wait();
    EXPECT_EQ(g_avg.size(), g_vecSize);
}

// 有小于等于0双入参标准差
TEST_F(StdCpuTest, test_CalcStd2_thredpool_2)
{
    m_tp->commit([]() {
        for (auto it = g_v0.begin(); it != g_v0.end(); it++)
        {
            g_std.push_back(std::move(common::cpu::CalcStd(*it, true)));
        }
    });
    m_tp->wait();
    EXPECT_EQ(g_std.size(), g_vecSize);
}

/*
[ RUN      ] StdCpuTest.test_CalcAvg2_thredpool_1
[       OK ] StdCpuTest.test_CalcAvg2_thredpool_1 (247 ms)
[ RUN      ] StdCpuTest.test_CalcStd2_thredpool_1
[       OK ] StdCpuTest.test_CalcStd2_thredpool_1 (474 ms)
[ RUN      ] StdCpuTest.test_CalcAvg2_thredpool_2
[       OK ] StdCpuTest.test_CalcAvg2_thredpool_2 (184 ms)
[ RUN      ] StdCpuTest.test_CalcStd2_thredpool_2
[       OK ] StdCpuTest.test_CalcStd2_thredpool_2 (354 ms)
小结：
1.这个量级线程池没有任何优势，反而变慢了
2.线程池中异步执行倒还好
*/

// 有小于等于0双入参均值
TEST_F(StdCpuTest, test_CalcAvg2_serial_1)
{
    common::cpu::CalcAvg(g_v0, true, g_avg);
    EXPECT_EQ(g_avg.size(), g_vecSize);
}

// 有小于等于0双入参标准差
TEST_F(StdCpuTest, test_CalcStd2_serial_1)
{
    common::cpu::CalcStd(g_v0, true, g_std);
    EXPECT_EQ(g_std.size(), g_vecSize);
}

// 有小于等于0双入参标准差
TEST_F(StdCpuTest, test_CalcAvgAndStd_serial_1)
{
    common::cpu::CalcAvgAndStd(g_v0, true, g_avg, g_std);
    EXPECT_EQ(g_avg.size(), g_vecSize);
    EXPECT_EQ(g_std.size(), g_vecSize);
    std::cout << "g_avg: "
              << common::to_string(std::vector<float>(g_avg.begin(), g_avg.begin() + 10))
              << std::endl;
    std::cout << "g_std: "
              << common::to_string(std::vector<float>(g_std.begin(), g_std.begin() + 10))
              << std::endl;
}

// 没有小于等于0双入参标准差
TEST_F(StdCpuTest, test_CalcAvgAndStd_serial_2)
{
    common::cpu::CalcAvgAndStd(g_v, false, g_avg, g_std);
    EXPECT_EQ(g_avg.size(), g_vecSize);
    EXPECT_EQ(g_std.size(), g_vecSize);
    std::cout << "g_avg: "
              << common::to_string(std::vector<float>(g_avg.begin(), g_avg.begin() + 10))
              << std::endl;
    std::cout << "g_std: "
              << common::to_string(std::vector<float>(g_std.begin(), g_std.begin() + 10))
              << std::endl;
}

/*
[ RUN      ] StdCpuTest.test_CalcAvg2_serial_1
[       OK ] StdCpuTest.test_CalcAvg2_serial_1 (175 ms)
[ RUN      ] StdCpuTest.test_CalcStd2_serial_1
[       OK ] StdCpuTest.test_CalcStd2_serial_1 (335 ms)
[ RUN      ] StdCpuTest.test_CalcAvgAndStd_serial_1
[       OK ] StdCpuTest.test_CalcAvgAndStd_serial_1 (335 ms)
小结：
1.这个量级并行接口与异步调用相当，略有提升，建议优先使用串行版本
*/

// 有小于等于0双入参均值
TEST_F(StdCpuTest, test_CalcAvg2_parallel_1)
{
    common::cpu::CalcAvgParallel(g_v0, true, g_avg);
    EXPECT_EQ(g_avg.size(), g_vecSize);
}

// 有小于等于0双入参标准差
TEST_F(StdCpuTest, test_CalcStd2_parallel_1)
{
    common::cpu::CalcStdParallel(g_v0, true, g_std);
    EXPECT_EQ(g_std.size(), g_vecSize);
}

// 有小于等于0双入参标准差
TEST_F(StdCpuTest, test_CalcAvgAndStd_parallel_1)
{
    common::cpu::CalcAvgAndStdParallel(g_v0, true, g_avg, g_std);
    EXPECT_EQ(g_avg.size(), g_vecSize);
    EXPECT_EQ(g_std.size(), g_vecSize);
}

/*
[ RUN      ] StdCpuTest.test_CalcAvg2_parallel_1
[       OK ] StdCpuTest.test_CalcAvg2_parallel_1 (175 ms)
[ RUN      ] StdCpuTest.test_CalcStd2_parallel_1
[       OK ] StdCpuTest.test_CalcStd2_parallel_1 (335 ms)
[ RUN      ] StdCpuTest.test_CalcAvgAndStd_parallel_1
[       OK ] StdCpuTest.test_CalcAvgAndStd_parallel_1 (335 ms)
小结：
1.这个量级并行接口与异步调用相当，建议优先使用串行版本
*/

TEST_F(StdCpuTest, others_1)
{
    std::cout << "sizeof(float):" << sizeof(float) << std::endl;
    std::cout << "sizeof(float32_t):" << sizeof(float32_t) << std::endl;
    std::cout << "sizeof(float64_t):" << sizeof(float64_t) << std::endl;
    std::cout << "sizeof(double):" << sizeof(double) << std::endl;
    std::cout << "sizeof(long double):" << sizeof(long double) << std::endl;
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
    printf("runing: %s g_vecSize[%lu] g_deqSize[%lu] \n", argv[0], g_vecSize, g_deqSize);

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
