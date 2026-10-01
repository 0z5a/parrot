# Optimized-build baseline findings

Test date: 2026-09-27. CUDA 13.0.88, CCCL 3.4.2, doctest 2.4.11,
RTX 5090. Compile flags include -std=c++20 -arch=sm_120 --extended-lambda -O2.

The complete current P2 suite returned 234/237 cases and 15,837/15,842
assertions passing. Its three failures are in existing array operations:
MinArrayTest, MaxArrayTest, CombinedArrayAndScalarTest. The reduction-only
suite, including the new stream and device-isolation tests, passed 28/28
cases and 15,048/15,048 assertions.

| Source / check | Passed cases | Passed assertions | Outcome |
|---|---:|---:|---|
| Original upstream d7d47a6, array operations at -O2 | 33/36 | 80/85 | Same three failures |
| P1 9356c16, array operations at -O2 | 33/36 | 80/85 | Same three failures |
| P2 6031fe5, full suite with extended tests at -O2 | 234/237 | 15,837/15,842 | Same three failures |
| Isolated upstream diagnostic described below | 36/36 | 85/85 | Passed |

The upstream archive's parrot.hpp Git blob is
`dd0d26782ec52a2be88b1daeeeb2b0e1b6f94014`, matching the pinned upstream commit.
The two scratch implementations also matched their published Git blobs.
The earlier full-suite binary passed 234/234 cases and still passes the
three selected cases; that result does not establish optimized-build correctness.

## Isolated diagnosis

The min/max functors return `decltype(a < b ? a : b)` and the corresponding
max expression from const-reference arguments. These are reference return
types. Returning such references through lazy transform/zip evaluation can
outlive converted temporary values. In an isolated copy of the original
upstream headers, changing both returns to `T` removed the unstable results.
The combined test also has an incorrect expected element: max([15,25,35,45],20)
then min([25,15,40,30]) is [20,15,35,30], not [20,15,40,30]. Correcting this
reference together with value returns passed all 36 array cases at -O2.

These diagnostic edits are not part of the stream/scratch production diff.
The baseline failure and proposed repair need independent review. No all-green
release-suite claim is made for this draft.
