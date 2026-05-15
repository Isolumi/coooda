# Milestone 01: Environment And Measurement

## Deliverable

Reusable CUDA checks, device info, timers, and benchmark helpers.

## Implement

- `coooda::core::DeviceInfo`
- `coooda::core::device_count()`
- `coooda::core::get_device_info(int device_id)`
- `coooda::core::check_cuda(...)`
- `coooda::core::check_last_kernel(...)`
- `coooda::core::synchronize_device()`
- `coooda::core::CpuTimer`
- `coooda::core::CudaEventTimer`
- `coooda::tests::close_enough(...)`
- `coooda::tests::FailureReport`
- `coooda::bench::BenchmarkConfig`
- `coooda::bench::TimingStats`
- `coooda::bench::run_benchmark(...)`

## Steps

1. Print CUDA device info.
2. Add CUDA call and kernel launch checking.
3. Add CPU timing.
4. Add CUDA event timing.
5. Add tolerance and failure-report helpers.
6. Add benchmark warmups, repeated trials, and median/min/max summaries.
7. Benchmark no-op launch, host/device copy, and device sync.

## Done

- A failed CUDA call reports file, line, and message.
- A benchmark report includes warmups, trials, median, min, and max.
- `notes/measurement.md` explains what each timer measures.
