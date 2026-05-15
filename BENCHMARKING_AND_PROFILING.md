# Benchmarking And Profiling

Benchmark only after correctness passes.

## Benchmark Report

Record:

- operation
- variant
- shape
- dtype
- build mode
- GPU
- warmups
- trials
- median/min/max time
- bandwidth or FLOP/s
- speedup versus baseline

## Profiling

Use Nsight only for claims that matter. If you say something is faster because of memory access, occupancy, fusion, or tiling, include profiler evidence.
