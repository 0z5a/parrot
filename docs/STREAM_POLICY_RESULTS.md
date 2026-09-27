# Explicit stream-ordered scratch: final validation

The revised P2 makes stream-ordered scratch **opt-in**. Existing calls use
the P1 synchronous allocation strategy. The public `parrot.hpp` is byte-for-byte
identical to P1, including synchronous result ownership. This avoids imposing
pool-allocation costs on applications that synchronize and read back each result.

```cpp
// Existing behavior, including the explicit-stream overload:
thrustx::reduce_by_n(first, last, out, width, op, init, stream);

// Opt-in for a preallocated pipeline with a nondefault stream:
thrustx::reduce_by_n(first, last, out, width, op, init, stream,
                     thrustx::scratch_allocation::stream_ordered);
```

The opt-in path queues allocation → CUB reduction → release on the same stream.
Input/output storage and the stream must remain alive until their final use
completes. A null stream retains synchronous scratch. The library does not
change pool attributes, create a global cache, or claim CUDA Graph support.

## Correctness and release-build status

RTX 5090, CUDA 13.0.88, CCCL 3.4.2, doctest 2.4.11, `-O2 -std=c++20 -arch=sm_120`.

| Final check | Result |
|---|---|
| Reduction suite, both allocation policies | 29/29 cases; 15,056/15,056 assertions |
| Independent producer/reduce/consumer stream chains | CPU reference passed for all saved outputs |
| Two-device isolation, 1,000 reductions/device | Passed |
| Compute Sanitizer memcheck with full leak check | 0 errors; 0 bytes leaked |
| Compute Sanitizer synccheck | 0 errors |
| Complete optimized suite | 235/238 cases; 15,845/15,850 assertions |

The same three existing array-operation failures remain: MinArrayTest,
MaxArrayTest, CombinedArrayAndScalarTest. They also fail on pinned upstream
and P1. See [release-build findings](RELEASE_TEST_FINDINGS.md). No all-green
full-suite claim is made, and the min/max repair is outside this PR.

## Default public-path performance

Each hardware ran six balanced orderings of P1, final P2 default, and P2 opt-in,
using fresh processes, 20 warmups and 200 timed iterations. The redundant
opt-in-binary public case was skipped because the public API uses the default
policy. There are **60 measured arms per hardware**, all reference checks passed.
CPU affinity: logical CPU 103 on the RTX host and CPU 0 on the A100 host.
Both are shared hosts; other jobs and clocks were left unchanged.

Public timing includes reduce/result allocation, stream completion,
per-element `to_host()`, host checking, and result destruction.

| Hardware | Segments × width | P1 median | Final P2 default | Latency change |
|---|---:|---:|---:|---:|
| RTX 5090 | 128 × 8 | 1,003.39 μs | 1,008.10 μs | +0.47% |
| RTX 5090 | 2,048 × 128 | 16,004.86 μs | 16,026.77 μs | +0.14% |
| A100 | 128 × 8 | 1,043.49 μs | 1,039.95 μs | −0.34% |
| A100 | 2,048 × 128 | 17,118.61 μs | 17,085.73 μs | −0.19% |

The previously observed 15.8–27.4% default public-path regression did not recur.
These small differences support comparable performance for the tested paths;
they are not a public-API speedup claim.

## Preallocated producer → reduce → consumer

Input/output allocation is outside the loop. Scratch allocation/free,
producer and consumer kernels, and final completion are included.

| Hardware | Segments × width | P1 median | P2 default | P2 opt-in |
|---|---:|---:|---:|---:|
| RTX 5090 | 128 × 8 | 27.75 μs | 22.74 μs | 22.58 μs |
| RTX 5090 | 2,048 × 128 | 24.39 μs | 23.38 μs | 21.56 μs |
| A100 | 128 × 8 | 16.44 μs | 15.41 μs | 17.30 μs |
| A100 | 2,048 × 128 | 18.09 μs | 18.17 μs | 20.26 μs |

The opt-in path is not universally faster. On A100 these small workloads
are slower with asynchronous scratch, which supports making the policy
explicit. RTX measurements include substantial outliers; all samples are
retained. Comparisons between P2 default and opt-in isolate the allocator
choice more directly than attributing every P1/P2 difference to it.

## Independence from an unrelated stream

`tests/bench_stream_independence.cu` submits a finite delay kernel on a
separate nonblocking stream, then measures host return from reduction.
The delay uses the device clock with a nominal 200 ms target; actual time
varies with clock behavior. Three fresh processes ran per policy and device.

| Hardware | Synchronous median host return | Stream-ordered median host return | Unrelated event after opt-in return |
|---|---:|---:|---|
| RTX 5090 | 164,908.75 μs | 272.51 μs | Still pending in all 3 runs |
| A100 | 200,002.54 μs | 834.74 μs | Still pending in all 3 runs |

All reductions produced the correct outputs. In every synchronous run the
unrelated event was already complete on return. This demonstrates avoided
host waiting under the tested independent-stream workload. It is not a
kernel speedup or total-application speed ratio. Every delay finished normally.

## Diagnosis and retained evidence

Stage measurements of the earlier automatic policy separated reduce,
synchronization, host copy, and destruction. The existing `to_host()`
performs one host transfer per element. A diagnostic-only retained-pool
configuration reduced synchronization costs in several runs, while other
jobs produced sizeable outliers.

CUDA's default pool release threshold is zero, and synchronization can release
unused pool memory back to the OS; see the
[NVIDIA CUDA 13 memory-pool API](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html).
The stage experiment supports pool churn as a contributing factor, not an
exclusive causal attribution for every observed slowdown. The production
change preserves caller control instead of adjusting a process's pool policy.

Historical automatic-policy results remain in
[extended results](EXTENDED_STREAM_RESULTS.md).
Final and diagnostic RTX samples, logs and hashes are in
`STREAM_POLICY_RTX5090.json`; A100 samples and hashes are in
`STREAM_POLICY_A100.json`.

## Reproduction

Use the existing toolchain and pinned CCCL includes from the earlier report.
For the chain benchmark, compile the P1 and P2-default arms without a macro;
compile the final P2 opt-in arm with `-DPARROT_STREAM_ORDERED_SCRATCH`.
Use `-arch=sm_120` on RTX 5090 and `-arch=sm_80` on A100. The same source
`tests/bench_stream_chain.cu` is used in all arms.

Name the binaries `builds/bench_optin_p1`, `builds/bench_optin_default`,
and `builds/bench_optin_async`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/run_stream_policy_benchmarks.py results.jsonl
CUDA_VISIBLE_DEVICES=0 builds/bench_stream_independence sync
CUDA_VISIBLE_DEVICES=0 builds/bench_stream_independence async
```

The earlier narrow `tests/bench_stream_reduction.cu` benchmark uses the same
macro to select explicit opt-in. Its final opt-in binary was also compiled
and smoke-tested. No model or dataset was downloaded for this work;
cross-host execution copied source code only.
