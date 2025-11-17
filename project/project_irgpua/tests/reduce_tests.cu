#include <gtest/gtest.h>

#include "reduce.cuh"

#define BLOCK_SIZE 256

TEST(ReduceTests, SingleElement)
{
    rmm::device_vector<int> input(1, 42);
    int result = reduce_byhand(input);
    EXPECT_EQ(result, 42);
}

TEST(ReduceTests, SumToZero) {
    rmm::device_vector<int> input{-1, 1, -1, 1, -1, 1};
    int result = reduce_byhand(input);
    EXPECT_EQ(result, 0);
}

TEST(ReduceTests, BlockSizeElements)
{
    rmm::device_vector<int> input(BLOCK_SIZE, 1);
    int result = reduce_byhand(input);
    EXPECT_EQ(result, BLOCK_SIZE);
}

TEST(ReduceTests, TenBlockSizeElements)
{
    rmm::device_vector<int> input(BLOCK_SIZE * 10, 1);
    int result = reduce_byhand(input);
    EXPECT_EQ(result, BLOCK_SIZE * 10);
}

TEST(ReduceTests, HundredBlockSizeElements)
{
    rmm::device_vector<int> input(BLOCK_SIZE * 100, 1);
    int result = reduce_byhand(input);
    EXPECT_EQ(result, BLOCK_SIZE * 100);
}
