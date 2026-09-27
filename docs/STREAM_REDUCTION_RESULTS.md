# Segmented reduction stream validation

Tested on one RTX 5090 (SM120), CUDA 13.0.88, CCCL 3.4.2, using a
nonblocking CUDA stream. The test records a producer event on one stream,
waits for it on the reduction stream, reduces two 4-element segments, and
consumes the result on that same stream. The expected outputs are 20 and 52.
The existing reduction suite plus this regression passed: 25 cases,
48 assertions. Full Parrot suite: 234 cases, 842 assertions passed. Targeted CUDA
`compute-sanitizer --tool memcheck` found 0 errors.

## Host-visible latency

`tests/bench_stream_reduction.cu` submits 500 reductions of 128 segments × 8 integers to
one nonblocking stream, then synchronizes. Both versions perform 20 warmup
calls. Ten paired runs were made on GPU 0; each row below is the median host
wall time per call, including enqueue and final synchronization. The input
and output allocations are outside the timed region.

| Implementation | Median time per call | Relative speed |
|---|---:|---:|
| Explicit stream with synchronous scratch allocation/free | 15.06 μs | 1.00× |
| Explicit stream with stream-ordered scratch allocation/free | 9.62 μs | 1.57× |

The original API has no explicit stream parameter, so it has no matching
nonblocking-stream result. Individual calls ranged from 13.29–22.47 μs
with synchronous scratch and 9.10–15.72 μs with stream-ordered scratch.
This is a small-kernel enqueue benchmark, not a claim about all reductions.
The default-stream path retains its existing allocator behavior. Full-array
reductions still return a host scalar and are not asynchronous.
