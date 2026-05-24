#include <coooda_cuda/nn/normalization.cuh>

#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>
#include <coooda_cuda/memory/device_buffer.cuh>

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

struct WelfordState {
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;
};

void require_positive_epsilon(float epsilon, const char *operation) {
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        coooda_core::fail(std::string(operation) + " requires positive finite epsilon");
    }
}

void require_norm_shape(
    std::size_t input_size,
    std::size_t gamma_size,
    std::size_t feature_count,
    float epsilon,
    const char *operation
) {
    if (feature_count == 0) {
        coooda_core::fail(std::string(operation) + " requires non-zero feature count");
    }
    if (gamma_size != feature_count) {
        coooda_core::fail(std::string(operation) + " gamma size must equal feature count");
    }
    if (input_size % feature_count != 0) {
        coooda_core::fail(std::string(operation) + " input size must be divisible by feature count");
    }
    require_positive_epsilon(epsilon, operation);
}

void require_layer_norm_shape(
    std::size_t input_size,
    std::size_t gamma_size,
    std::size_t beta_size,
    std::size_t feature_count,
    float epsilon,
    const char *operation
) {
    require_norm_shape(input_size, gamma_size, feature_count, epsilon, operation);
    if (beta_size != feature_count) {
        coooda_core::fail(std::string(operation) + " beta size must equal feature count");
    }
}

int checked_count(std::size_t size, const char *operation) {
    if (size > static_cast<std::size_t>(INT_MAX)) {
        coooda_core::fail(std::string(operation) + " input is too large for the CUDA normalization kernel");
    }
    return static_cast<int>(size);
}

__device__ WelfordState welford_add(WelfordState state, float value) {
    state.count += 1;
    const float delta = value - state.mean;
    state.mean += delta / static_cast<float>(state.count);
    const float delta2 = value - state.mean;
    state.m2 += delta * delta2;
    return state;
}

__device__ WelfordState welford_combine(WelfordState a, WelfordState b) {
    if (a.count == 0) {
        return b;
    }
    if (b.count == 0) {
        return a;
    }

    const int count = a.count + b.count;
    const float delta = b.mean - a.mean;
    const float count_f = static_cast<float>(count);
    const float a_count = static_cast<float>(a.count);
    const float b_count = static_cast<float>(b.count);

    WelfordState combined;
    combined.count = count;
    combined.mean = a.mean + (delta * b_count / count_f);
    combined.m2 = a.m2 + b.m2 + ((delta * delta) * a_count * b_count / count_f);
    return combined;
}

__global__ void layer_norm_kernel(
    const float *input,
    const float *gamma,
    const float *beta,
    float *out,
    int feature_count,
    float epsilon
) {
    __shared__ float values[kBlockSize];

    const int row = blockIdx.x;
    const int row_offset = row * feature_count;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < feature_count; col += blockDim.x) {
        sum += input[row_offset + col];
    }
    values[threadIdx.x] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            values[threadIdx.x] += values[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float mean = values[0] / static_cast<float>(feature_count);

    float variance_sum = 0.0f;
    for (int col = threadIdx.x; col < feature_count; col += blockDim.x) {
        const float centered = input[row_offset + col] - mean;
        variance_sum += centered * centered;
    }
    values[threadIdx.x] = variance_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            values[threadIdx.x] += values[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float variance = values[0] / static_cast<float>(feature_count);
    const float inv_std = rsqrtf(variance + epsilon);

    for (int col = threadIdx.x; col < feature_count; col += blockDim.x) {
        out[row_offset + col] = ((input[row_offset + col] - mean) * inv_std * gamma[col]) + beta[col];
    }
}

__global__ void layer_norm_welford_kernel(
    const float *input,
    const float *gamma,
    const float *beta,
    float *out,
    int feature_count,
    float epsilon
) {
    __shared__ WelfordState states[kBlockSize];

    const int row = blockIdx.x;
    const int row_offset = row * feature_count;

    WelfordState local;
    for (int col = threadIdx.x; col < feature_count; col += blockDim.x) {
        local = welford_add(local, input[row_offset + col]);
    }

    states[threadIdx.x] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            states[threadIdx.x] = welford_combine(states[threadIdx.x], states[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    const WelfordState row_state = states[0];
    const float variance = row_state.m2 / static_cast<float>(feature_count);
    const float inv_std = rsqrtf(variance + epsilon);

    for (int col = threadIdx.x; col < feature_count; col += blockDim.x) {
        out[row_offset + col] = ((input[row_offset + col] - row_state.mean) * inv_std * gamma[col]) + beta[col];
    }
}

__global__ void rms_norm_kernel(
    const float *input,
    const float *gamma,
    float *out,
    int feature_count,
    float epsilon
) {
    __shared__ float values[kBlockSize];

    const int row = blockIdx.x;
    const int row_offset = row * feature_count;

    float square_sum = 0.0f;
    for (int col = threadIdx.x; col < feature_count; col += blockDim.x) {
        const float value = input[row_offset + col];
        square_sum += value * value;
    }
    values[threadIdx.x] = square_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            values[threadIdx.x] += values[threadIdx.x + offset];
        }
        __syncthreads();
    }

    const float mean_square = values[0] / static_cast<float>(feature_count);
    const float inv_rms = rsqrtf(mean_square + epsilon);
    for (int col = threadIdx.x; col < feature_count; col += blockDim.x) {
        out[row_offset + col] = input[row_offset + col] * inv_rms * gamma[col];
    }
}

} // namespace

namespace coooda_cuda::nn {

std::vector<float> layer_norm_baseline(
    const std::vector<float> &input,
    const std::vector<float> &gamma,
    const std::vector<float> &beta,
    std::size_t feature_count,
    float epsilon
) {
    require_layer_norm_shape(
        input.size(),
        gamma.size(),
        beta.size(),
        feature_count,
        epsilon,
        "layer_norm_baseline"
    );
    if (input.empty()) {
        return {};
    }

    const int features = checked_count(feature_count, "layer_norm_baseline");
    const int rows = checked_count(input.size() / feature_count, "layer_norm_baseline");
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_gamma = coooda_cuda::memory::DeviceBuffer::from_host(gamma);
    coooda_cuda::memory::DeviceBuffer device_beta = coooda_cuda::memory::DeviceBuffer::from_host(beta);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    layer_norm_kernel<<<rows, kBlockSize>>>(
        device_input.data(),
        device_gamma.data(),
        device_beta.data(),
        device_out.data(),
        features,
        epsilon
    );
    COODA_CUDA_CHECK_LAST("layer_norm_baseline");
    return device_out.copy_to_host();
}

std::vector<float> layer_norm_welford(
    const std::vector<float> &input,
    const std::vector<float> &gamma,
    const std::vector<float> &beta,
    std::size_t feature_count,
    float epsilon
) {
    require_layer_norm_shape(
        input.size(),
        gamma.size(),
        beta.size(),
        feature_count,
        epsilon,
        "layer_norm_welford"
    );
    if (input.empty()) {
        return {};
    }

    const int features = checked_count(feature_count, "layer_norm_welford");
    const int rows = checked_count(input.size() / feature_count, "layer_norm_welford");
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_gamma = coooda_cuda::memory::DeviceBuffer::from_host(gamma);
    coooda_cuda::memory::DeviceBuffer device_beta = coooda_cuda::memory::DeviceBuffer::from_host(beta);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    layer_norm_welford_kernel<<<rows, kBlockSize>>>(
        device_input.data(),
        device_gamma.data(),
        device_beta.data(),
        device_out.data(),
        features,
        epsilon
    );
    COODA_CUDA_CHECK_LAST("layer_norm_welford");
    return device_out.copy_to_host();
}

std::vector<float> rms_norm_baseline(
    const std::vector<float> &input,
    const std::vector<float> &gamma,
    std::size_t feature_count,
    float epsilon
) {
    require_norm_shape(input.size(), gamma.size(), feature_count, epsilon, "rms_norm_baseline");
    if (input.empty()) {
        return {};
    }

    const int features = checked_count(feature_count, "rms_norm_baseline");
    const int rows = checked_count(input.size() / feature_count, "rms_norm_baseline");
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_gamma = coooda_cuda::memory::DeviceBuffer::from_host(gamma);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    rms_norm_kernel<<<rows, kBlockSize>>>(
        device_input.data(),
        device_gamma.data(),
        device_out.data(),
        features,
        epsilon
    );
    COODA_CUDA_CHECK_LAST("rms_norm_baseline");
    return device_out.copy_to_host();
}

} // namespace coooda_cuda::nn
