# Build System

Milestone 00 creates the build system. Keep it simple.

## Required Targets

- `coooda`: library target
- `coooda_tests`: correctness tests
- `coooda_bench`: benchmarks
- `coooda_tiny_gpt`: final app

## Required Build Modes

- Debug: warnings, assertions, CUDA debug info
- Release: optimized benchmarks
- Profile: release-like build suitable for Nsight

## Commands To Maintain

Document your exact commands for:

- configure debug
- configure release
- build
- run tests
- run one benchmark
- run Nsight Systems
- run Nsight Compute
- clean build output

Do not move to Milestone 01 until an empty test and empty benchmark target both run.
