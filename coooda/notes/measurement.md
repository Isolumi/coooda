# Measurement Notes

`coooda::core::CpuTimer` measures host wall-clock time with
`std::chrono::steady_clock`. Use it for CPU work or for CUDA work that includes
an explicit device synchronization inside the measured region.

`coooda::core::CudaEventTimer` records CUDA events on a stream and reports GPU
elapsed time between those events. It measures device work queued between
`start()` and `stop_ms()` on that stream, not unrelated host-side setup.

`coooda::bench::run_benchmark` runs warmups first, then records repeated timed
trials. By default it synchronizes before and after each trial so asynchronous
CUDA launches are included in the measured host-side elapsed time. Disable those
synchronizations only when the callable already performs the synchronization or
when benchmarking host-only code.

The initial benchmark executable reports:

- no-op CUDA launch timing
- 16 MiB host-to-device copy timing
- device synchronization timing

Each report includes operation, variant, shape, dtype, build mode, GPU, warmups,
trials, median, min, max, and mean time.
