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
#include "thrustx.hpp"

void check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}
__global__ void delay_cycles(unsigned long long cycles) {
    auto start = clock64();
    while (clock64() - start < cycles) {}
}
int main(int argc, char **argv) {
    bool async = argc > 1 && std::string(argv[1]) == "async";
    auto policy = async ? thrustx::scratch_allocation::stream_ordered
                        : thrustx::scratch_allocation::synchronous;
    cudaStream_t work, unrelated;
    cudaEvent_t done;
    int *input, *output, clock_khz;
    check(cudaMalloc(&input, 8 * sizeof(int)));
    check(cudaMalloc(&output, 2 * sizeof(int)));
    check(cudaStreamCreateWithFlags(&work, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&unrelated, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    check(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    check(cudaMemsetAsync(input, 1, 8 * sizeof(int), work));
    for (int i = 0; i < 20; ++i) {
        thrustx::reduce_by_n(input, input + 8, output, 4, cuda::std::plus<int>{}, 0, work, policy);
    }
    check(cudaStreamSynchronize(work));
    delay_cycles<<<1, 1, 0, unrelated>>>(static_cast<unsigned long long>(clock_khz) * 200);
    check(cudaGetLastError());
    check(cudaEventRecord(done, unrelated));
    auto start = std::chrono::steady_clock::now();
    thrustx::reduce_by_n(input, input + 8, output, 4, cuda::std::plus<int>{}, 0, work, policy);
    auto stop = std::chrono::steady_clock::now();
    auto state = cudaEventQuery(done);
    if (state != cudaSuccess && state != cudaErrorNotReady) { check(state); }
    check(cudaStreamSynchronize(work));
    int host[2];
    check(cudaMemcpy(host, output, sizeof(host), cudaMemcpyDeviceToHost));
    bool correct = host[0] == 4 * 0x01010101 && host[1] == 4 * 0x01010101;
    std::printf("{\"policy\":\"%s\",\"host_return_us\":%.3f,"
                "\"unrelated_pending\":%s,\"correct\":%s}\n",
                async ? "stream_ordered" : "synchronous",
                std::chrono::duration<double, std::micro>(stop-start).count(),
                state == cudaErrorNotReady ? "true" : "false", correct ? "true" : "false");
    check(cudaStreamSynchronize(unrelated));
    check(cudaFree(input)); check(cudaFree(output));
    check(cudaEventDestroy(done));
    check(cudaStreamDestroy(work)); check(cudaStreamDestroy(unrelated));
    return correct ? 0 : 1;
}
