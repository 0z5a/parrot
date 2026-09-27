# Extended stream validation and application-path timing

Hardware/toolchain: RTX 5090, CUDA 13.0.88, CCCL 3.4.2, doctest 2.4.11.
The producer/reduction/consumer and device-isolation tests use two visible GPUs.
Headers match published P1 `9356c163958b1009ca5b382fae61580120cda11c`
and P2 `6031fe532ee91c5b8a9a5c1961b6b8326f19895c`.

## Correctness

The extended reduction suite passed **28 cases / 15,048 assertions** with P2.
It covers two independent dependency chains, each with separate producer,
reduction, and consumer nonblocking streams. Widths 1, 7, 32, and 255 each
run 200 iterations with distinct inputs, a nonzero initial value, and a
CPU reference for every saved output. Events establish both forward
dependencies and safe buffer reuse. No device-wide synchronization is used
inside the chain.

Further checks cover empty inputs, invalid/nondividing segment widths,
column-wise public API forwarding, and 1,000 interleaved reductions per
device on two devices. Each device has independent allocations and streams.

| P2 check | Result |
|---|---|
| Extended reduction suite, optimized build | 28/28 cases; 15,048/15,048 assertions |
| Compute Sanitizer memcheck, leak-check full | 0 errors; 0 bytes leaked |
| Compute Sanitizer synccheck | 0 errors |
| Entire optimized suite | 234/237 cases; three pre-existing array-operation failures |

The complete optimized suite is **not green**. The same three array tests
fail on the pinned original upstream and P1. See
[release-build findings](RELEASE_TEST_FINDINGS.md) for the exact comparison
and isolated min/max reference-return diagnosis. The earlier 234-case pass
does not establish release-build correctness.

## End-to-end paths

`tests/bench_stream_chain.cu` measures two workloads:

- `chain`: GPU producer → segmented reduction → GPU consumer → final
  stream completion. Input/output allocations are outside the timed loop;
  per-reduction scratch allocation/free remains inside it. The producer
  generates signed integers and final results are checked against a CPU sum.
- `public`: `range(...).reshape(...).reduce(..., stream)` → stream completion
  → `to_host()` → host correctness check, including result allocation and
  destruction each iteration. Input construction is outside the timed loop.
  The existing `to_host()` reads each element separately.

Both use 20 warmups followed by 200 iterations. Each arm starts a fresh
process. Eight paired comparisons per shape alternate P1/P2 ordering.
The confirmation run pins the processes to logical CPU 103 and uses GPU 0.
Other users' workloads remain on the shared host; neither clocks nor the
environment were changed.

| CPU-pinned workload | Segments × width | P1 median | P2 median | P1/P2 | Interpretation |
|---|---:|---:|---:|---:|---|
| Producer → reduce → consumer | 128 × 8 | 24.66 μs | 20.47 μs | 1.20× | Unstable; one P2 arm was 479.81 μs |
| Producer → reduce → consumer | 2,048 × 128 | 24.62 μs | 20.75 μs | 1.19× | Improvement in this preallocated path |
| Public API → host results | 128 × 8 | 1,138.14 μs | 1,450.31 μs | 0.785× | P2 latency increased 27.4% |
| Public API → host results | 2,048 × 128 | 17,987.42 μs | 20,831.56 μs | 0.863× | P2 latency increased 15.8% |

The public-path regression also appeared in the earlier unpinned run:
983.31 → 1,357.94 μs and 16,836.66 → 20,556.20 μs. That run's preallocated
chain medians were 27.21 → 21.81 μs and 23.71 → 20.57 μs.
Both runs and every outlier are retained in the raw data. CPU affinity
reduced some variation but did not remove the public-path regression.

A paired log-ratio bootstrap (8 pairs, 10,000 resamples, seed 20260927)
gives the following geometric speed ratios and nominal 95% intervals:

| CPU-pinned path | Geometric ratio | Interval |
|---|---:|---:|
| Chain, 128 × 8 | 0.876× | 0.371–1.459× |
| Chain, 2,048 × 128 | 1.252× | 1.182–1.366× |
| Public, 128 × 8 | 0.676× | 0.551–0.792× |
| Public, 2,048 × 128 | 0.842× | 0.794–0.886× |

These intervals describe the observed pairs on a shared host, not hardware
independence or a universal performance guarantee. The small-chain outlier
prevents a reliable speedup claim for that confirmation case.
`stream_timeline_us` includes stream idle time between host submissions;
it is not kernel-only duration. For `public`, `submit_us` also includes
the synchronizations and host copies inside each iteration.

**The earlier 1.57× narrow scratch benchmark does not establish a speedup
for the public API. The measured public-path slowdown remains unresolved,
and this PR stays draft.** No production implementation was changed during
this follow-up.

## Build and evidence

Example commands from the workspace containing `src/parrot` and the
pinned `deps` directories:

```bash
nvcc -std=c++20 -arch=sm_120 --extended-lambda -O2 \
  -I src/parrot -I deps/cccl-3.4.2/thrust -I deps/cccl-3.4.2/cub \
  -I deps/cccl-3.4.2/libcudacxx/include \
  src/parrot/tests/bench_stream_chain.cu -o builds/bench_chain_p2
CUDA_VISIBLE_DEVICES=0 taskset -c 103 builds/bench_chain_p2 chain 2048 128
CUDA_VISIBLE_DEVICES=0 taskset -c 103 builds/bench_chain_p2 public 128 8
CUDA_VISIBLE_DEVICES=0,1 compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 builds/test_extended_p2
CUDA_VISIBLE_DEVICES=0,1 compute-sanitizer --tool synccheck \
  --error-exitcode 99 builds/test_extended_p2
```

Build the P1 arm with the same source/flags but point the first include path
at the pinned P1 headers. `docs/stream_followup_data.json` contains all
128 measured arms, summaries, hashes, and test output.
