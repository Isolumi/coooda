#pragma once

#include <cuda_runtime.h>

#include <chrono>

namespace coooda::core {

class CpuTimer {
  public:
    void start();
    double stop_ms();
    double elapsed_ms() const;
    bool running() const noexcept;

  private:
    using Clock = std::chrono::steady_clock;

    Clock::time_point start_time_{};
    double elapsed_ms_ = 0.0;
    bool running_ = false;
};

class CudaEventTimer {
  public:
    CudaEventTimer();
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer &) = delete;
    CudaEventTimer &operator=(const CudaEventTimer &) = delete;

    void start(cudaStream_t stream = nullptr);
    float stop_ms(cudaStream_t stream = nullptr);
    float elapsed_ms() const;
    bool running() const noexcept;

  private:
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
    float elapsed_ms_ = 0.0f;
    bool running_ = false;
    bool has_elapsed_ = false;
};

} // namespace coooda::core
