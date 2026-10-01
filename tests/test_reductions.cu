/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <limits>
#include "parrot.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_common.hpp"

// Test deltas function
TEST_CASE("ParrotTest - DeltasTest") {
    auto arr    = parrot::array({1, 3, 6, 10});
    auto result = arr.deltas().sum();
    CHECK_EQ(result.value(), 9);  // sum of 2,3,4
}

// Test maxr function
TEST_CASE("ParrotTest - MaxrTest") {
    auto arr    = parrot::array({1, 5, 3, 2});
    auto result = arr.maxr();
    CHECK_EQ(result.value(), 5);
}

// Test minr function
TEST_CASE("ParrotTest - MinrTest") {
    auto arr    = parrot::array({1, 5, 3, 2});
    auto result = arr.minr();
    CHECK_EQ(result.value(), 1);
}

// Test minr with empty array (using only initial value)
TEST_CASE("ParrotTest - MinrEmptyTest") {
    auto arr    = parrot::array<int>({});
    auto result = arr.minr();
    CHECK_EQ(result.value(), std::numeric_limits<int>::max());
}

// Test minr with negative values
TEST_CASE("ParrotTest - MinrNegativeTest") {
    auto arr    = parrot::array({-1, -5, 3, 2});
    auto result = arr.minr();
    CHECK_EQ(result.value(), -5);
}

// Test minmax function
TEST_CASE("ParrotTest - MinmaxTest") {
    auto arr    = parrot::array({3, 1, 7, 5, 2});
    auto result = arr.minmax().to_host();
    REQUIRE_EQ(result.size(), 1);
    CHECK_EQ(result[0].first, 1);   // minimum value
    CHECK_EQ(result[0].second, 7);  // maximum value
}

// Test minmax with negative values
TEST_CASE("ParrotTest - MinmaxNegativeTest") {
    auto arr    = parrot::array({-3, 1, -7, 5, 2});
    auto result = arr.minmax().to_host();
    REQUIRE_EQ(result.size(), 1);
    CHECK_EQ(result[0].first, -7);  // minimum value
    CHECK_EQ(result[0].second, 5);  // maximum value
}

// Test any() method with all zeros
TEST_CASE("ParrotTest - AnyAllZerosTest") {
    auto arr    = parrot::array({0, 0, 0, 0});
    auto result = arr.any();
    CHECK_FALSE(result.value());
}

// Test any() method with some non-zeros
TEST_CASE("ParrotTest - AnySomeNonZerosTest") {
    auto arr    = parrot::array({0, 0, 3, 0});
    auto result = arr.any();
    CHECK(result.value());
}

// Test all() method with all non-zeros
TEST_CASE("ParrotTest - AllNonZerosTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.all();
    CHECK(result.value());
}

// Test all() method with some zeros
TEST_CASE("ParrotTest - AllSomeZerosTest") {
    auto arr    = parrot::array({1, 0, 3, 4});
    auto result = arr.all();
    CHECK_FALSE(result.value());
}

// Test any() and all() with empty array
TEST_CASE("ParrotTest - AnyAllEmptyTest") {
    auto arr = parrot::array<int>({});
    CHECK_FALSE(
      arr.any().value());  // Empty array should return false for any()
    CHECK(
      arr.all()
        .value());  // Empty array should return true for all() (vacuously true)
}

// Test prod function
TEST_CASE("ParrotTest - ProdTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.prod();
    CHECK_EQ(result.value(), 24);  // product of 1,2,3,4 is 24
}

// Test prod with an empty array
TEST_CASE("ParrotTest - ProdEmptyTest") {
    auto arr    = parrot::array<int>({});
    auto result = arr.prod();
    CHECK_EQ(result.value(),
             1);  // product of an empty array is the identity (1)
}

// Test prod with floating point values
TEST_CASE("ParrotTest - ProdFloatTest") {
    auto arr    = parrot::array<float>({1.5F, 2.0F, 2.5F});
    auto result = arr.prod();
    CHECK(result.value() ==
          doctest::Approx(7.5F));  // product of 1.5*2.0*2.5 = 7.5
}

// Test reduce function with plus operation
TEST_CASE("ParrotTest - ReducePlusTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.reduce(0, parrot::add{});
    CHECK_EQ(result.value(), 10);
    CHECK_EQ(result.value(),
             arr.sum().value());  // Verify it matches the sum() function
}

// Test reduce function with multiplies operation
TEST_CASE("ParrotTest - ReduceMultipliesTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.reduce(1, parrot::mul{});
    CHECK_EQ(result.value(), 24);
    CHECK_EQ(result.value(),
             arr.prod().value());  // Verify it matches the prod() function
}

// Test reduce function with maximum operation
TEST_CASE("ParrotTest - ReduceMaximumTest") {
    auto arr    = parrot::array({1, 5, 3, 4});
    auto result = arr.reduce(std::numeric_limits<int>::lowest(), parrot::max{});
    CHECK_EQ(result.value(), 5);
    CHECK_EQ(result.value(),
             arr.maxr().value());  // Verify it matches the maxr() function
}

// Test reduce function with minimum operation
TEST_CASE("ParrotTest - ReduceMinimumTest") {
    auto arr    = parrot::array({5, 2, 3, 4});
    auto result = arr.reduce(std::numeric_limits<int>::max(), parrot::min{});
    CHECK_EQ(result.value(), 2);
    CHECK_EQ(result.value(),
             arr.minr().value());  // Verify it matches the minr() function
}

// Test stats::mode function
TEST_CASE("ParrotTest - StatsModeTest") {
    auto arr    = parrot::array({3, 1, 3, 1, 2, 3});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), 3);  // 3 appears most frequently (3 times)
}

// Test stats::mode with single mode
TEST_CASE("ParrotTest - StatsModeSingleTest") {
    auto arr    = parrot::array({1, 2, 2, 3});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), 2);  // 2 appears most frequently (2 times)
}

// Test stats::mode with all unique elements
TEST_CASE("ParrotTest - StatsModeUniqueTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), 1);  // All elements appear once, returns smallest
}

// Test stats::mode with single element
TEST_CASE("ParrotTest - StatsModeSingleElementTest") {
    auto arr    = parrot::array({42});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), 42);  // Single element is the mode
}

// Test stats::mode with negative numbers
TEST_CASE("ParrotTest - StatsModeNegativeTest") {
    auto arr    = parrot::array({-1, -2, -1, -3, -1});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), -1);  // -1 appears most frequently (3 times)
}
__global__ void fill_stream_input(int *values, int offset) {
    int i = threadIdx.x;
    if (i < 8) { values[i] = i + offset; }
}

__global__ void scale_stream_output(int *values) {
    int i = threadIdx.x;
    if (i < 2) { values[i] *= 2; }
}

TEST_CASE("ParrotTest - ExplicitStreamSegmentedReduction") {
    thrust::device_vector<int> input(8, thrust::default_init);
    thrust::device_vector<int> output(2, thrust::default_init);
    cudaStream_t producer, reduction;
    cudaEvent_t ready;
    REQUIRE_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking),
               cudaSuccess);
    REQUIRE_EQ(cudaStreamCreateWithFlags(&reduction, cudaStreamNonBlocking),
               cudaSuccess);
    REQUIRE_EQ(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
               cudaSuccess);

    fill_stream_input<<<1, 32, 0, producer>>>(
      thrust::raw_pointer_cast(input.data()), 1);
    REQUIRE_EQ(cudaEventRecord(ready, producer), cudaSuccess);
    REQUIRE_EQ(cudaStreamWaitEvent(reduction, ready), cudaSuccess);
    thrustx::reduce_by_n(input.begin(),
                         input.end(),
                         output.begin(),
                         4,
                         cuda::std::plus<int>{},
                         0,
                         reduction);
    scale_stream_output<<<1, 32, 0, reduction>>>(
      thrust::raw_pointer_cast(output.data()));
    REQUIRE_EQ(cudaStreamSynchronize(reduction), cudaSuccess);
    CHECK_EQ(output[0], 20);
    CHECK_EQ(output[1], 52);

    auto matrix = parrot::array({1, 2, 3, 4, 5, 6, 7, 8}).reshape({2, 4});
    auto sums   = matrix.reduce(
      0, parrot::add{}, std::integral_constant<int, 2>{}, reduction);
    REQUIRE_EQ(cudaStreamSynchronize(reduction), cudaSuccess);
    auto host = sums.to_host();
    CHECK_EQ(host[0], 10);
    CHECK_EQ(host[1], 26);
    CHECK_THROWS_AS(
      (void)matrix.reduce(
        0, parrot::add{}, std::integral_constant<int, 0>{}, reduction),
      std::invalid_argument);

    REQUIRE_EQ(cudaEventDestroy(ready), cudaSuccess);
    REQUIRE_EQ(cudaStreamDestroy(reduction), cudaSuccess);
    REQUIRE_EQ(cudaStreamDestroy(producer), cudaSuccess);
}

__global__ void fill_stress_input(int *values, int count, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { values[i] = (i % 17) - 8 + seed; }
}

__global__ void store_stress_output(const int *values, int *history, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { history[i] = 2 * values[i]; }
}

TEST_CASE("ParrotTest - IndependentStreamDependencyChains") {
    constexpr int steps = 200;
    for (int width : {1, 7, 32, 255}) {
        constexpr int segments = 37;
        const int count = width * segments;
        int *inputs[2], *outputs[2], *history[2];
        cudaStream_t producer[2], reduction[2], consumer[2];
        cudaEvent_t ready[2], reduced[2], consumed[2];
        for (int lane = 0; lane < 2; ++lane) {
            REQUIRE_EQ(cudaMalloc(&inputs[lane], count * sizeof(int)), cudaSuccess);
            REQUIRE_EQ(cudaMalloc(&outputs[lane], segments * sizeof(int)), cudaSuccess);
            REQUIRE_EQ(cudaMalloc(&history[lane], steps * segments * sizeof(int)), cudaSuccess);
            REQUIRE_EQ(cudaStreamCreateWithFlags(&producer[lane], cudaStreamNonBlocking), cudaSuccess);
            REQUIRE_EQ(cudaStreamCreateWithFlags(&reduction[lane], cudaStreamNonBlocking), cudaSuccess);
            REQUIRE_EQ(cudaStreamCreateWithFlags(&consumer[lane], cudaStreamNonBlocking), cudaSuccess);
            REQUIRE_EQ(cudaEventCreateWithFlags(&ready[lane], cudaEventDisableTiming), cudaSuccess);
            REQUIRE_EQ(cudaEventCreateWithFlags(&reduced[lane], cudaEventDisableTiming), cudaSuccess);
            REQUIRE_EQ(cudaEventCreateWithFlags(&consumed[lane], cudaEventDisableTiming), cudaSuccess);
        }
        for (int step = 0; step < steps; ++step) {
            for (int lane = 0; lane < 2; ++lane) {
                if (step) {
                    REQUIRE_EQ(cudaStreamWaitEvent(producer[lane], consumed[lane]), cudaSuccess);
                }
                fill_stress_input<<<(count + 127) / 128, 128, 0, producer[lane]>>>(
                    inputs[lane], count, step + lane * 1000);
                REQUIRE_EQ(cudaGetLastError(), cudaSuccess);
                REQUIRE_EQ(cudaEventRecord(ready[lane], producer[lane]), cudaSuccess);
                REQUIRE_EQ(cudaStreamWaitEvent(reduction[lane], ready[lane]), cudaSuccess);
                thrustx::reduce_by_n(inputs[lane], inputs[lane] + count, outputs[lane],
                                     width, cuda::std::plus<int>{}, 3, reduction[lane],
                                     thrustx::scratch_allocation::stream_ordered);
                REQUIRE_EQ(cudaEventRecord(reduced[lane], reduction[lane]), cudaSuccess);
                REQUIRE_EQ(cudaStreamWaitEvent(consumer[lane], reduced[lane]), cudaSuccess);
                store_stress_output<<<1, 128, 0, consumer[lane]>>>(
                    outputs[lane], history[lane] + step * segments, segments);
                REQUIRE_EQ(cudaGetLastError(), cudaSuccess);
                REQUIRE_EQ(cudaEventRecord(consumed[lane], consumer[lane]), cudaSuccess);
            }
        }
        for (int lane = 0; lane < 2; ++lane) {
            REQUIRE_EQ(cudaStreamSynchronize(consumer[lane]), cudaSuccess);
            std::vector<int> host(steps * segments);
            REQUIRE_EQ(cudaMemcpy(host.data(), history[lane], host.size() * sizeof(int),
                                   cudaMemcpyDeviceToHost), cudaSuccess);
            bool matches = true;
            for (int step = 0; step < steps; ++step) {
                for (int segment = 0; segment < segments; ++segment) {
                    int expected = 3;
                    for (int j = 0; j < width; ++j) {
                        expected += ((segment * width + j) % 17) - 8 + step + lane * 1000;
                    }
                    matches &= host[step * segments + segment] == 2 * expected;
                }
            }
            CHECK(matches);
            REQUIRE_EQ(cudaFree(inputs[lane]), cudaSuccess);
            REQUIRE_EQ(cudaFree(outputs[lane]), cudaSuccess);
            REQUIRE_EQ(cudaFree(history[lane]), cudaSuccess);
            REQUIRE_EQ(cudaEventDestroy(ready[lane]), cudaSuccess);
            REQUIRE_EQ(cudaEventDestroy(reduced[lane]), cudaSuccess);
            REQUIRE_EQ(cudaEventDestroy(consumed[lane]), cudaSuccess);
            REQUIRE_EQ(cudaStreamDestroy(producer[lane]), cudaSuccess);
            REQUIRE_EQ(cudaStreamDestroy(reduction[lane]), cudaSuccess);
            REQUIRE_EQ(cudaStreamDestroy(consumer[lane]), cudaSuccess);
        }
    }
}

TEST_CASE("ParrotTest - ExplicitStreamBoundariesAndColumns") {
    thrust::device_vector<int> input(8, 1), output(8, -123);
    cudaStream_t stream;
    REQUIRE_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    CHECK_NOTHROW(thrustx::reduce_by_n(input.begin(), input.begin(), output.begin(),
                                      4, cuda::std::plus<int>{}, 0, stream));
    for (int width : {0, -1, 3}) {
        CHECK_THROWS_AS(thrustx::reduce_by_n(input.begin(), input.end(), output.begin(),
                                            width, cuda::std::plus<int>{}, 0, stream),
                        std::invalid_argument);
    }
    REQUIRE_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    CHECK_EQ(output[0], -123);
    auto matrix = parrot::array({1, 2, 3, 4, 5, 6, 7, 8}).reshape({2, 4});
    auto columns = matrix.reduce(0, parrot::add{}, std::integral_constant<int, 1>{}, stream);
    REQUIRE_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    auto host = columns.to_host();
    for (int i = 0; i < 4; ++i) { CHECK_EQ(host[i], 6 + 2 * i); }
    REQUIRE_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_CASE("ParrotTest - TwoDeviceScratchIsolation") {
    int devices;
    REQUIRE_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
    if (devices < 2) {
        MESSAGE("TwoDeviceScratchIsolation requires two visible GPUs");
        return;
    }
    constexpr int steps = 1000, segments = 64, width = 8, count = segments * width;
    int *input[2], *output[2];
    cudaStream_t stream[2];
    int original_device;
    REQUIRE_EQ(cudaGetDevice(&original_device), cudaSuccess);
    for (int device = 0; device < 2; ++device) {
        REQUIRE_EQ(cudaSetDevice(device), cudaSuccess);
        REQUIRE_EQ(cudaMalloc(&input[device], count * sizeof(int)), cudaSuccess);
        REQUIRE_EQ(cudaMalloc(&output[device], steps * segments * sizeof(int)), cudaSuccess);
        REQUIRE_EQ(cudaStreamCreateWithFlags(&stream[device], cudaStreamNonBlocking), cudaSuccess);
        fill_stress_input<<<4, 128, 0, stream[device]>>>(input[device], count, device * 100);
        REQUIRE_EQ(cudaGetLastError(), cudaSuccess);
    }
    for (int step = 0; step < steps; ++step) {
        for (int device = 0; device < 2; ++device) {
            REQUIRE_EQ(cudaSetDevice(device), cudaSuccess);
            thrustx::reduce_by_n(input[device], input[device] + count,
                output[device] + step * segments, width, cuda::std::plus<int>{}, 0, stream[device],
                thrustx::scratch_allocation::stream_ordered);
        }
    }
    for (int device = 0; device < 2; ++device) {
        REQUIRE_EQ(cudaSetDevice(device), cudaSuccess);
        REQUIRE_EQ(cudaStreamSynchronize(stream[device]), cudaSuccess);
        std::vector<int> host(steps * segments);
        REQUIRE_EQ(cudaMemcpy(host.data(), output[device], host.size() * sizeof(int),
                               cudaMemcpyDeviceToHost), cudaSuccess);
        bool matches = true;
        for (int step = 0; step < steps; ++step) {
            for (int segment = 0; segment < segments; ++segment) {
                int expected = 0;
                for (int j = 0; j < width; ++j) {
                    expected += ((segment * width + j) % 17) - 8 + device * 100;
                }
                matches &= host[step * segments + segment] == expected;
            }
        }
        CHECK(matches);
        REQUIRE_EQ(cudaFree(input[device]), cudaSuccess);
        REQUIRE_EQ(cudaFree(output[device]), cudaSuccess);
        REQUIRE_EQ(cudaStreamDestroy(stream[device]), cudaSuccess);
    }
    REQUIRE_EQ(cudaSetDevice(original_device), cudaSuccess);
}


TEST_CASE("ParrotTest - ExplicitScratchAllocationPolicies") {
    thrust::device_vector<int> input(8, 2), output(2, 0);
    cudaStream_t stream;
    REQUIRE_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    for (auto policy : {thrustx::scratch_allocation::synchronous,
                        thrustx::scratch_allocation::stream_ordered}) {
        thrustx::reduce_by_n(input.begin(), input.end(), output.begin(), 4,
                            cuda::std::plus<int>{}, 3, stream, policy);
        REQUIRE_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        CHECK_EQ(output[0], 11);
        CHECK_EQ(output[1], 11);
    }
    REQUIRE_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
