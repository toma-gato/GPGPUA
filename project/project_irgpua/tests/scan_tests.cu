#include <gtest/gtest.h>

#include "scan.cuh"

#define BLOCK_SIZE 256

TEST(ScanTests, SingleElement)
{
    rmm::device_vector<int> input(1, 42);
    rmm::device_vector<int> output(2, 0);

    exclusive_scan_byhand(input, output);

    std::vector<int> h_output(output.size());
    thrust::copy(output.begin(), output.end(), h_output.begin());

    std::vector<int> expected{0, 42};

    EXPECT_EQ(h_output, expected);
}

TEST(ScanTests, TenElements)
{
    rmm::device_vector<int> input(10, 1);
    rmm::device_vector<int> output(11, 0);

    exclusive_scan_byhand(input, output);

    std::vector<int> h_output(output.size());
    thrust::copy(output.begin(), output.end(), h_output.begin());

    std::vector<int> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    EXPECT_EQ(h_output, expected);
}

TEST(ScanTests, BlockSizeElements)
{
    rmm::device_vector<int> input(BLOCK_SIZE, 1);
    rmm::device_vector<int> output(BLOCK_SIZE + 1, 0);

    exclusive_scan_byhand(input, output);

    std::vector<int> h_output(output.size());
    thrust::copy(output.begin(), output.end(), h_output.begin());

    std::vector<int> expected(BLOCK_SIZE + 1);
    for (int i = 0; i <= BLOCK_SIZE; ++i)
    {
        expected[i] = i;
    }

    EXPECT_EQ(h_output, expected);
}

TEST(ScanTests, TenBlockSizeElements)
{
    rmm::device_vector<int> input(BLOCK_SIZE * 10, 1);
    rmm::device_vector<int> output(BLOCK_SIZE * 10 + 1, 0);

    exclusive_scan_byhand(input, output);

    std::vector<int> h_output(output.size());
    thrust::copy(output.begin(), output.end(), h_output.begin());

    std::vector<int> expected(BLOCK_SIZE * 10 + 1);
    for (int i = 0; i <= BLOCK_SIZE * 10; ++i)
    {
        expected[i] = i;
    }

    EXPECT_EQ(h_output, expected);
}

TEST(ScanTests, HundredBlockSizeElements)
{
    rmm::device_vector<int> input(BLOCK_SIZE * 100, 1);
    rmm::device_vector<int> output(BLOCK_SIZE * 100 + 1, 0);

    exclusive_scan_byhand(input, output);

    std::vector<int> h_output(output.size());
    thrust::copy(output.begin(), output.end(), h_output.begin());

    std::vector<int> expected(BLOCK_SIZE * 100 + 1);
    for (int i = 0; i <= BLOCK_SIZE * 100; ++i)
    {
        expected[i] = i;
    }

    EXPECT_EQ(h_output, expected);
}
