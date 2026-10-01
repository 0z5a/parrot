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

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include "thrustx.hpp"

struct add_int {
    __host__ __device__ int operator()(int a, int b) const { return a + b; }
};

int main() {
    constexpr int segments = 128;
    constexpr int width = 8;
    constexpr int repetitions = 500;
    int *input, *output;
    cudaStream_t stream;
    cudaMalloc(&input, segments * width * sizeof(int));
    cudaMalloc(&output, segments * sizeof(int));
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaMemsetAsync(input, 1, segments * width * sizeof(int), stream);
    for (int i = 0; i < 20; ++i) {
#ifdef PARROT_STREAM_ORDERED_SCRATCH
        thrustx::reduce_by_n(input, input + segments * width, output, width,
                             add_int{}, 0, stream, thrustx::scratch_allocation::stream_ordered);
#else
        thrustx::reduce_by_n(input, input + segments * width, output, width,
                             add_int{}, 0, stream);
#endif
    }
    cudaStreamSynchronize(stream);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
#ifdef PARROT_STREAM_ORDERED_SCRATCH
        thrustx::reduce_by_n(input, input + segments * width, output, width,
                             add_int{}, 0, stream, thrustx::scratch_allocation::stream_ordered);
#else
        thrustx::reduce_by_n(input, input + segments * width, output, width,
                             add_int{}, 0, stream);
#endif
    }
    const auto status = cudaStreamSynchronize(stream);
    const auto stop = std::chrono::steady_clock::now();
    int first;
    cudaMemcpy(&first, output, sizeof(int), cudaMemcpyDeviceToHost);
    const double us = std::chrono::duration<double, std::micro>(stop - start).count() / repetitions;
    std::printf("status=%d first=%d us_per_call=%.3f\n", status, first, us);
    cudaStreamDestroy(stream);
    cudaFree(output);
    cudaFree(input);
    return status == cudaSuccess && first == 8 * 0x01010101 ? 0 : 1;
}
