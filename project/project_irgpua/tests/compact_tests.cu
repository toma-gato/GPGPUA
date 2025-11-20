#include <gtest/gtest.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <algorithm>
#include <iterator>

#include "compact.cuh"

TEST(CompactTests, SingleElementWithFlag)
{
    constexpr int flag = -27;
    rmm::device_vector<int> input{-27};
    rmm::device_vector<int> output(1);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}

TEST(CompactTests, SingleElementWithoutFlag)
{
    constexpr int flag = 42;
    rmm::device_vector<int> input{-27};
    rmm::device_vector<int> output(1);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}

TEST(CompactTests, TenElementsWithFlag)
{
    constexpr int flag = -27;
    rmm::device_vector<int> input{1, -27, 3, -27, 5, 6, -27, 8, 9, 10};
    rmm::device_vector<int> output(10);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}

TEST(CompactTests, TenElementsWithoutFlag)
{
    constexpr int flag = 99;
    rmm::device_vector<int> input{1, -27, 3, -27, 5, 6, -27, 8, 9, 10};
    rmm::device_vector<int> output(10);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}

TEST(CompactTests, BlockSizeElementsWithFlag)
{
    constexpr int flag = -1;
    constexpr size_t block_size = 256;
    rmm::device_vector<int> input(block_size);
    for (size_t i = 0; i < block_size; ++i)
    {
        input[i] = (i % 3 == 0) ? flag : static_cast<int>(i);
    }
    rmm::device_vector<int> output(block_size);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}

TEST(CompactTests, BlockSizeElementsWithoutFlag)
{
    constexpr int flag = -999;
    constexpr size_t block_size = 256;
    rmm::device_vector<int> input(block_size, 1);
    rmm::device_vector<int> output(block_size);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}

TEST(CompactTests, TenBlockSizeElementsWithFlag)
{
    constexpr int flag = 0;
    constexpr size_t block_size = 256;
    constexpr size_t num_blocks = 10;
    rmm::device_vector<int> input(block_size * num_blocks);
    for (size_t i = 0; i < block_size * num_blocks; ++i)
    {
        input[i] = (i % 5 == 0) ? flag : static_cast<int>(i);
    }
    rmm::device_vector<int> output(block_size * num_blocks);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}

TEST(CompactTests, TenBlockSizeElementsWithoutFlag)
{
    constexpr int flag = -1000;
    constexpr size_t block_size = 256;
    constexpr size_t num_blocks = 10;
    rmm::device_vector<int> input(block_size * num_blocks, 1);
    rmm::device_vector<int> output(block_size * num_blocks);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}

TEST(CompactTests, HundredBlockSizeElementsWithFlag)
{
    constexpr int flag = 7;
    constexpr size_t block_size = 256;
    constexpr size_t num_blocks = 100;
    rmm::device_vector<int> input(block_size * num_blocks);
    for (size_t i = 0; i < block_size * num_blocks; ++i)
    {
        input[i] = (i % 7 == 0) ? flag : static_cast<int>(i);
    }
    rmm::device_vector<int> output(block_size * num_blocks);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}


TEST(CompactTests, DenseRegionsOfFlag)
{
    constexpr int flag = -27;
    // Large dense regions of `flag` separated by small non-flag spans
    constexpr size_t region1 = 2048;
    constexpr size_t region2 = 4096;
    constexpr size_t region3 = 1024;
    constexpr size_t sep = 8;
    constexpr size_t tail = 16;

    const size_t total = region1 + sep + region2 + sep + region3 + tail;
    rmm::device_vector<int> input(total);
    size_t idx = 0;

    for (size_t i = 0; i < region1; ++i)
        input[idx++] = flag;
    for (size_t i = 0; i < sep; ++i)
        input[idx++] = static_cast<int>(i + 1); // non-flag
    for (size_t i = 0; i < region2; ++i)
        input[idx++] = flag;
    for (size_t i = 0; i < sep; ++i)
        input[idx++] = static_cast<int>(100 + i); // non-flag
    for (size_t i = 0; i < region3; ++i)
        input[idx++] = flag;
    for (size_t i = 0; i < tail; ++i)
        input[idx++] = static_cast<int>(500 + i); // non-flag tail

    rmm::device_vector<int> output(total);

    compact_byhand(input, output, flag);

    thrust::host_vector<int> h_input = input;
    std::vector<int> expected;
    expected.reserve(h_input.size());
    std::copy_if(h_input.begin(), h_input.end(), std::back_inserter(expected),
                 [&flag](int x)
                 { return x != flag; });
    expected.resize(expected.size());

    thrust::host_vector<int> h_output = output;
    EXPECT_EQ(h_output, expected);
}


