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
#include <string>
#include "parrot.hpp"

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}
__global__ void produce_chain(int *input, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { input[i] = i % 17 - 8; }
}
__global__ void consume_chain(int *output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { output[i] *= 2; }
}
int main(int argc, char **argv) {
    if (argc != 4) { return 2; }
    std::string mode = argv[1];
    int segments = std::stoi(argv[2]), width = std::stoi(argv[3]);
    int count = segments * width;
    constexpr int repetitions = 200, warmup = 20;
    cudaStream_t stream;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    thrust::device_vector<int> input(count, thrust::default_init);
    thrust::device_vector<int> output(segments, thrust::default_init);
    auto *in = thrust::raw_pointer_cast(input.data());
    auto *out = thrust::raw_pointer_cast(output.data());
    auto matrix = parrot::range(count).reshape({segments, width});
    bool correct = true;
    auto call = [&] {
        if (mode == "chain") {
            produce_chain<<<(count + 255) / 256, 256, 0, stream>>>(in, count);
            check_cuda(cudaGetLastError());
#ifdef PARROT_STREAM_ORDERED_SCRATCH
            thrustx::reduce_by_n(in, in + count, out, width, cuda::std::plus<int>{}, 0,
                                 stream, thrustx::scratch_allocation::stream_ordered);
#else
            thrustx::reduce_by_n(in, in + count, out, width, cuda::std::plus<int>{}, 0, stream);
#endif
            consume_chain<<<(segments + 255) / 256, 256, 0, stream>>>(out, segments);
            check_cuda(cudaGetLastError());
        } else {
            auto result = matrix.reduce(0, parrot::add{}, std::integral_constant<int, 2>{}, stream);
            check_cuda(cudaStreamSynchronize(stream));
            auto host = result.to_host();
            for (int s = 0; s < segments; ++s) {
                int expected = width * (s * width + 1) + width * (width - 1) / 2;
                correct &= host[s] == expected;
            }
        }
    };
    for (int i = 0; i < warmup; ++i) { call(); }
    check_cuda(cudaStreamSynchronize(stream));
    cudaEvent_t start, end;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&end));
    auto t0 = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(start, stream));
    for (int i = 0; i < repetitions; ++i) { call(); }
    auto submitted = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(end, stream));
    check_cuda(cudaEventSynchronize(end));
    auto t1 = std::chrono::steady_clock::now();
    float elapsed_ms;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, end));
    if (mode == "chain") {
        std::vector<int> host(segments);
        check_cuda(cudaMemcpy(host.data(), out, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < segments; ++s) {
            int expected = 0;
            for (int j = 0; j < width; ++j) { expected += (s * width + j) % 17 - 8; }
            correct &= host[s] == 2 * expected;
        }
    }
    double total_us = std::chrono::duration<double, std::micro>(t1-t0).count() / repetitions;
    double submit_us = std::chrono::duration<double, std::micro>(submitted-t0).count() / repetitions;
    std::printf("{\"mode\":\"%s\",\"segments\":%d,\"width\":%d,\"total_us\":%.4f,"
                "\"submit_us\":%.4f,\"stream_timeline_us\":%.4f,\"correct\":%s}\n",
                mode.c_str(), segments, width, total_us, submit_us, elapsed_ms * 1000 / repetitions,
                correct ? "true" : "false");
    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(end));
    check_cuda(cudaStreamDestroy(stream));
    return correct ? 0 : 1;
}
