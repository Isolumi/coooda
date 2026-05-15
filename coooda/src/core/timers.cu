#include <coooda/core/timers.hpp>

#include <coooda/core/cuda_checks.hpp>

#include <stdexcept>

namespace coooda::core {

void CpuTimer::start() {
    start_time_ = Clock::now();
    elapsed_ms_ = 0.0;
    running_ = true;
}

double CpuTimer::stop_ms() {
    if (!running_) {
        throw std::logic_error("CpuTimer::stop_ms called before start");
    }

    const Clock::time_point stop_time = Clock::now();
    elapsed_ms_ = std::chrono::duration<double, std::milli>(stop_time - start_time_).count();
    running_ = false;
    return elapsed_ms_;
}

double CpuTimer::elapsed_ms() const {
    if (running_) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_time_).count();
    }
    return elapsed_ms_;
}

bool CpuTimer::running() const noexcept { return running_; }

CudaEventTimer::CudaEventTimer() {
    COODA_CHECK_CUDA(cudaEventCreate(&start_event_));
    try {
        COODA_CHECK_CUDA(cudaEventCreate(&stop_event_));
    } catch (...) {
        cudaEventDestroy(start_event_);
        start_event_ = nullptr;
        throw;
    }
}

CudaEventTimer::~CudaEventTimer() {
    if (stop_event_ != nullptr) {
        cudaEventDestroy(stop_event_);
    }
    if (start_event_ != nullptr) {
        cudaEventDestroy(start_event_);
    }
}

void CudaEventTimer::start(
    cudaStream_t stream
) {
    COODA_CHECK_CUDA(cudaEventRecord(start_event_, stream));
    running_ = true;
    has_elapsed_ = false;
    elapsed_ms_ = 0.0f;
}

float CudaEventTimer::stop_ms(
    cudaStream_t stream
) {
    if (!running_) {
        throw std::logic_error("CudaEventTimer::stop_ms called before start");
    }

    COODA_CHECK_CUDA(cudaEventRecord(stop_event_, stream));
    COODA_CHECK_CUDA(cudaEventSynchronize(stop_event_));
    COODA_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms_, start_event_, stop_event_));
    running_ = false;
    has_elapsed_ = true;
    return elapsed_ms_;
}

float CudaEventTimer::elapsed_ms() const {
    if (!has_elapsed_) {
        throw std::logic_error("CudaEventTimer::elapsed_ms called before stop_ms");
    }
    return elapsed_ms_;
}

bool CudaEventTimer::running() const noexcept { return running_; }

} // namespace coooda::core
