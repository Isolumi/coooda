# Coooda Scaffold Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current verbose milestone-only course with a buildable CMake starter scaffold and one-file CPU-to-CUDA checkpoint course units.

**Architecture:** The root project becomes a CMake workspace with four package libraries: `coooda_core`, `coooda_cpp`, `coooda_cuda`, and `coooda_compare`. C++ and CUDA backends remain separate packages, while shared test, benchmark, tensor, status, and comparison utilities live in `coooda_core` and `coooda_compare`.

**Tech Stack:** CMake 3.24+, C++17, CUDA C++17, CTest, standard library only, CUDA runtime only.

---

## File Structure

Create or replace these top-level files:

- Create: `CMakeLists.txt`
- Create: `CMakePresets.json`
- Create: `.gitignore`
- Modify: `README.md`
- Delete: `BENCHMARKING_AND_PROFILING.md`
- Delete: `BUILD_SYSTEM.md`
- Delete: `LIBRARY_ARCHITECTURE.md`
- Delete: `TESTING_AND_VALIDATION.md`
- Delete: `MILESTONE_00_PROJECT_SETUP_AND_CMAKE.md` through `MILESTONE_17_TINY_GPT_TRANSFORMER_CAPSTONE.md`
- Delete: `templates/`
- Delete: legacy tracked nested implementation tree `coooda/`

Create these package files:

```text
packages/coooda_core/CMakeLists.txt
packages/coooda_core/include/coooda_core/benchmark.hpp
packages/coooda_core/include/coooda_core/cuda_check.cuh
packages/coooda_core/include/coooda_core/device.hpp
packages/coooda_core/include/coooda_core/status.hpp
packages/coooda_core/include/coooda_core/tensor.hpp
packages/coooda_core/include/coooda_core/test_harness.hpp
packages/coooda_core/src/benchmark.cpp
packages/coooda_core/src/cuda_check.cu
packages/coooda_core/src/device.cu
packages/coooda_core/src/status.cpp
packages/coooda_core/src/tensor.cpp
packages/coooda_core/src/test_harness.cpp

packages/coooda_cpp/CMakeLists.txt
packages/coooda_cpp/include/coooda_cpp/ops/vector.hpp
packages/coooda_cpp/include/coooda_cpp/package.hpp
packages/coooda_cpp/src/ops/vector.cpp
packages/coooda_cpp/src/package.cpp

packages/coooda_cuda/CMakeLists.txt
packages/coooda_cuda/include/coooda_cuda/ops/vector.cuh
packages/coooda_cuda/include/coooda_cuda/package.cuh
packages/coooda_cuda/src/ops/vector.cu
packages/coooda_cuda/src/package.cu

packages/coooda_compare/CMakeLists.txt
packages/coooda_compare/include/coooda_compare/ops/vector_compare.hpp
packages/coooda_compare/include/coooda_compare/package.hpp
packages/coooda_compare/src/ops/vector_compare.cu
packages/coooda_compare/src/package.cpp
```

Create these test, benchmark, and app files:

```text
tests/CMakeLists.txt
tests/cpp/test_cpp_smoke.cpp
tests/cuda/test_cuda_smoke.cu
tests/compare/test_compare_smoke.cu

bench/CMakeLists.txt
bench/cpp/bench_cpp_smoke.cpp
bench/cuda/bench_cuda_smoke.cu
bench/compare/bench_compare_smoke.cu

apps/CMakeLists.txt
apps/inspect_device.cu
apps/run_checkpoint.cpp
apps/compare_backends.cu
apps/tiny_gpt_cpp.cpp
apps/tiny_gpt_cuda.cu
```

Create these course files:

```text
course/00_start_here.md
course/01_project_scaffold.md
course/02_core_helpers.md
course/03_memory_and_tensors.md
course/04_vector_ops.md
course/05_reductions.md
course/06_matmul.md
course/07_elementwise_and_fusion.md
course/08_softmax_and_loss.md
course/09_normalization.md
course/10_embeddings.md
course/11_linear_and_mlp.md
course/12_backward_and_optimizers.md
course/13_tiny_mlp_training.md
course/14_attention.md
course/15_transformer_block.md
course/16_tiny_gpt.md
```

---

### Task 1: Remove Legacy Course Surface And Add Root Project Metadata

**Files:**
- Delete: root milestone docs and old support docs listed in File Structure.
- Delete: `templates/`
- Delete: tracked legacy nested tree `coooda/`
- Create: `.gitignore`
- Create: `CMakePresets.json`
- Modify: `README.md`

- [ ] **Step 1: Stage old tracked scaffold and old docs for removal**

Run:

```bash
git rm --ignore-unmatch \
  BENCHMARKING_AND_PROFILING.md \
  BUILD_SYSTEM.md \
  LIBRARY_ARCHITECTURE.md \
  TESTING_AND_VALIDATION.md \
  MILESTONE_*.md
git rm --ignore-unmatch -r templates coooda
```

Expected: deleted files are staged. This is intentional because the approved redesign replaces the old milestone-only surface and the legacy nested implementation tree.

- [ ] **Step 2: Create `.gitignore`**

Write exactly:

```gitignore
build/
cmake-build-*/
.cache/
.DS_Store
compile_commands.json
*.ncu-rep
*.nsys-rep
*.qdrep
```

- [ ] **Step 3: Create `CMakePresets.json`**

Write exactly:

```json
{
  "version": 5,
  "configurePresets": [
    {
      "name": "debug",
      "displayName": "Debug",
      "generator": "Unix Makefiles",
      "binaryDir": "${sourceDir}/build/debug",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    },
    {
      "name": "release",
      "displayName": "Release",
      "generator": "Unix Makefiles",
      "binaryDir": "${sourceDir}/build/release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    },
    {
      "name": "profile",
      "displayName": "Profile",
      "generator": "Unix Makefiles",
      "binaryDir": "${sourceDir}/build/profile",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    }
  ],
  "buildPresets": [
    { "name": "debug", "configurePreset": "debug" },
    { "name": "release", "configurePreset": "release" },
    { "name": "profile", "configurePreset": "profile" }
  ],
  "testPresets": [
    {
      "name": "debug",
      "configurePreset": "debug",
      "output": { "outputOnFailure": true }
    }
  ]
}
```

- [ ] **Step 4: Replace `README.md`**

Write a short README with these sections:

```markdown
# Coooda

Coooda is a guided CUDA/C++ course that builds two parallel Transformer libraries:

- `coooda_cpp`: a pure C++ reference implementation.
- `coooda_cuda`: a CUDA implementation with baseline and optimized kernels.
- `coooda_compare`: runners that compare both backends.
- `coooda_core`: shared shapes, test helpers, benchmark helpers, CUDA checks, and reporting.

The course is organized as small checkpoints. Each topic keeps the CPU and CUDA work in the same file so the flow is always:

1. Write the C++ reference.
2. Test it.
3. Translate it to CUDA.
4. Compare CUDA against C++.
5. Optimize only after correctness passes.

## Quick Start

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
./build/debug/apps/inspect_device
./build/debug/apps/run_checkpoint
./build/debug/apps/compare_backends
```

Start at `course/00_start_here.md`.

## Project Layout

```text
packages/coooda_core/      shared infrastructure
packages/coooda_cpp/       C++ backend
packages/coooda_cuda/      CUDA backend
packages/coooda_compare/   backend comparison utilities
tests/                     CTest smoke and checkpoint tests
bench/                     benchmark runners
apps/                      runnable demos and final Tiny GPT apps
course/                    checkpoint guide
```
```

- [ ] **Step 5: Commit metadata cleanup**

Run:

```bash
git add .gitignore CMakePresets.json README.md
git status --short
git commit -m "chore: reset course project surface"
```

Expected: commit includes only legacy removals plus `.gitignore`, `CMakePresets.json`, and `README.md`.

---

### Task 2: Add Root CMake Workspace And Package Targets

**Files:**
- Create: `CMakeLists.txt`
- Create: `packages/coooda_core/CMakeLists.txt`
- Create: `packages/coooda_cpp/CMakeLists.txt`
- Create: `packages/coooda_cuda/CMakeLists.txt`
- Create: `packages/coooda_compare/CMakeLists.txt`

- [ ] **Step 1: Create root `CMakeLists.txt`**

Write exactly:

```cmake
cmake_minimum_required(VERSION 3.24)

project(coooda LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(coooda_options INTERFACE)
target_compile_options(coooda_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-lineinfo>
)

add_subdirectory(packages/coooda_core)
add_subdirectory(packages/coooda_cpp)
add_subdirectory(packages/coooda_cuda)
add_subdirectory(packages/coooda_compare)

foreach(coooda_optional_dir tests bench apps)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${coooda_optional_dir}/CMakeLists.txt")
    add_subdirectory(${coooda_optional_dir})
  endif()
endforeach()
```

- [ ] **Step 2: Create `packages/coooda_core/CMakeLists.txt`**

Write exactly:

```cmake
add_library(coooda_core
  src/benchmark.cpp
  src/cuda_check.cu
  src/device.cu
  src/status.cpp
  src/tensor.cpp
  src/test_harness.cpp
)

target_include_directories(coooda_core PUBLIC include)
target_link_libraries(coooda_core PUBLIC CUDA::cudart coooda_options)
```

- [ ] **Step 3: Create `packages/coooda_cpp/CMakeLists.txt`**

Write exactly:

```cmake
add_library(coooda_cpp
  src/ops/vector.cpp
  src/package.cpp
)

target_include_directories(coooda_cpp PUBLIC include)
target_link_libraries(coooda_cpp PUBLIC coooda_core coooda_options)
```

- [ ] **Step 4: Create `packages/coooda_cuda/CMakeLists.txt`**

Write exactly:

```cmake
add_library(coooda_cuda
  src/ops/vector.cu
  src/package.cu
)

target_include_directories(coooda_cuda PUBLIC include)
target_link_libraries(coooda_cuda PUBLIC coooda_core CUDA::cudart coooda_options)
```

- [ ] **Step 5: Create `packages/coooda_compare/CMakeLists.txt`**

Write exactly:

```cmake
add_library(coooda_compare
  src/ops/vector_compare.cu
  src/package.cpp
)

target_include_directories(coooda_compare PUBLIC include)
target_link_libraries(coooda_compare PUBLIC coooda_core coooda_cpp coooda_cuda coooda_options)
```

- [ ] **Step 6: Configure to verify CMake parses package skeleton**

Run:

```bash
cmake --preset debug
```

Expected: configure fails because source files do not exist yet. The failure should name the missing files from this task, not a CMake syntax error.

---

### Task 3: Implement `coooda_core` Boilerplate

**Files:**
- Create: `packages/coooda_core/include/coooda_core/status.hpp`
- Create: `packages/coooda_core/include/coooda_core/tensor.hpp`
- Create: `packages/coooda_core/include/coooda_core/test_harness.hpp`
- Create: `packages/coooda_core/include/coooda_core/benchmark.hpp`
- Create: `packages/coooda_core/include/coooda_core/cuda_check.cuh`
- Create: `packages/coooda_core/include/coooda_core/device.hpp`
- Create: `packages/coooda_core/src/status.cpp`
- Create: `packages/coooda_core/src/tensor.cpp`
- Create: `packages/coooda_core/src/test_harness.cpp`
- Create: `packages/coooda_core/src/benchmark.cpp`
- Create: `packages/coooda_core/src/cuda_check.cu`
- Create: `packages/coooda_core/src/device.cu`

- [ ] **Step 1: Add `status.hpp` and `status.cpp`**

`status.hpp`:

```cpp
#pragma once

#include <stdexcept>
#include <string>

namespace coooda_core {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string &message);
};

[[noreturn]] void fail(const std::string &message);

} // namespace coooda_core
```

`status.cpp`:

```cpp
#include <coooda_core/status.hpp>

namespace coooda_core {

Error::Error(const std::string &message) : std::runtime_error(message) {}

void fail(const std::string &message) {
    throw Error(message);
}

} // namespace coooda_core
```

- [ ] **Step 2: Add `tensor.hpp` and `tensor.cpp`**

`tensor.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace coooda_core {

struct Shape {
    std::vector<std::size_t> dims;
};

std::size_t numel(const Shape &shape);
std::string to_string(const Shape &shape);

} // namespace coooda_core
```

`tensor.cpp`:

```cpp
#include <coooda_core/tensor.hpp>

#include <numeric>
#include <sstream>

namespace coooda_core {

std::size_t numel(const Shape &shape) {
    if (shape.dims.empty()) {
        return 0;
    }

    return std::accumulate(
        shape.dims.begin(),
        shape.dims.end(),
        static_cast<std::size_t>(1),
        [](std::size_t acc, std::size_t dim) { return acc * dim; }
    );
}

std::string to_string(const Shape &shape) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.dims.size(); ++i) {
        if (i != 0) {
            out << "x";
        }
        out << shape.dims[i];
    }
    out << "]";
    return out.str();
}

} // namespace coooda_core
```

- [ ] **Step 3: Add test harness**

`test_harness.hpp`:

```cpp
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace coooda_core::test {

using TestFn = std::function<void()>;

struct TestCase {
    std::string name;
    TestFn run;
};

void require(bool condition, const std::string &message);
void require_equal(const std::string &expected, const std::string &actual, const std::string &message);
int run_tests(const std::vector<TestCase> &tests);

} // namespace coooda_core::test
```

`test_harness.cpp`:

```cpp
#include <coooda_core/test_harness.hpp>

#include <exception>
#include <iostream>
#include <stdexcept>

namespace coooda_core::test {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(const std::string &expected, const std::string &actual, const std::string &message) {
    if (expected != actual) {
        throw std::runtime_error(message + ": expected '" + expected + "', got '" + actual + "'");
    }
}

int run_tests(const std::vector<TestCase> &tests) {
    int failures = 0;
    for (const TestCase &test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << "\n";
        } catch (const std::exception &error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << "\n";
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << tests.size() << " test(s) passed\n";
    return 0;
}

} // namespace coooda_core::test
```

- [ ] **Step 4: Add benchmark helper**

`benchmark.hpp`:

```cpp
#pragma once

#include <chrono>
#include <functional>
#include <string>

namespace coooda_core::bench {

struct BenchmarkResult {
    std::string name;
    double elapsed_ms;
};

BenchmarkResult time_once(const std::string &name, const std::function<void()> &fn);
void print_result(const BenchmarkResult &result);

} // namespace coooda_core::bench
```

`benchmark.cpp`:

```cpp
#include <coooda_core/benchmark.hpp>

#include <iostream>

namespace coooda_core::bench {

BenchmarkResult time_once(const std::string &name, const std::function<void()> &fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    return BenchmarkResult{name, elapsed.count()};
}

void print_result(const BenchmarkResult &result) {
    std::cout << result.name << ": " << result.elapsed_ms << " ms\n";
}

} // namespace coooda_core::bench
```

- [ ] **Step 5: Add CUDA checks and device info**

`cuda_check.cuh`:

```cpp
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace coooda_core::cuda {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const std::string &message);
    cudaError_t code() const;

private:
    cudaError_t code_;
};

void check(cudaError_t code, const char *expr, const char *file, int line);
void check_last_kernel(const char *kernel, const char *file, int line);

} // namespace coooda_core::cuda

#define COODA_CUDA_CHECK(expr) ::coooda_core::cuda::check((expr), #expr, __FILE__, __LINE__)
#define COODA_CUDA_CHECK_LAST(kernel) ::coooda_core::cuda::check_last_kernel((kernel), __FILE__, __LINE__)
```

`cuda_check.cu`:

```cpp
#include <coooda_core/cuda_check.cuh>

#include <sstream>

namespace coooda_core::cuda {

CudaError::CudaError(cudaError_t code, const std::string &message)
    : std::runtime_error(message), code_(code) {}

cudaError_t CudaError::code() const {
    return code_;
}

void check(cudaError_t code, const char *expr, const char *file, int line) {
    if (code == cudaSuccess) {
        return;
    }

    std::ostringstream out;
    out << file << ":" << line << " CUDA call failed: " << expr << ": " << cudaGetErrorString(code);
    throw CudaError(code, out.str());
}

void check_last_kernel(const char *kernel, const char *file, int line) {
    check(cudaGetLastError(), kernel, file, line);
}

} // namespace coooda_core::cuda
```

`device.hpp`:

```cpp
#pragma once

#include <string>

namespace coooda_core::cuda {

struct DeviceInfo {
    int id;
    std::string name;
    std::size_t global_memory_bytes;
    int warp_size;
};

int device_count();
DeviceInfo get_device_info(int device_id);
std::string device_summary();

} // namespace coooda_core::cuda
```

`device.cu`:

```cpp
#include <coooda_core/device.hpp>
#include <coooda_core/cuda_check.cuh>

#include <cuda_runtime.h>

#include <sstream>

namespace coooda_core::cuda {

int device_count() {
    int count = 0;
    COODA_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

DeviceInfo get_device_info(int device_id) {
    cudaDeviceProp prop{};
    COODA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return DeviceInfo{device_id, prop.name, prop.totalGlobalMem, prop.warpSize};
}

std::string device_summary() {
    const int count = device_count();
    if (count == 0) {
        return "No CUDA devices found.";
    }

    const DeviceInfo info = get_device_info(0);
    std::ostringstream out;
    out << "CUDA device 0: " << info.name
        << ", global memory bytes=" << info.global_memory_bytes
        << ", warp size=" << info.warp_size;
    return out.str();
}

} // namespace coooda_core::cuda
```

- [ ] **Step 6: Commit core boilerplate**

Run:

```bash
git add packages/coooda_core CMakeLists.txt packages/coooda_core/CMakeLists.txt
git commit -m "feat: add core scaffold utilities"
```

Expected: commit contains only root CMake and `coooda_core`.

---

### Task 4: Add C++ CUDA And Compare Package Stubs

**Files:**
- Create: all `packages/coooda_cpp/`, `packages/coooda_cuda/`, and `packages/coooda_compare/` files listed in File Structure.

- [ ] **Step 1: Add C++ package**

`packages/coooda_cpp/include/coooda_cpp/package.hpp`:

```cpp
#pragma once

#include <string>

namespace coooda_cpp {

std::string backend_name();

} // namespace coooda_cpp
```

`packages/coooda_cpp/src/package.cpp`:

```cpp
#include <coooda_cpp/package.hpp>

namespace coooda_cpp {

std::string backend_name() {
    return "coooda_cpp";
}

} // namespace coooda_cpp
```

`packages/coooda_cpp/include/coooda_cpp/ops/vector.hpp`:

```cpp
#pragma once

#include <vector>

namespace coooda_cpp::ops {

std::vector<float> vector_add_reference(const std::vector<float> &a, const std::vector<float> &b);

} // namespace coooda_cpp::ops
```

`packages/coooda_cpp/src/ops/vector.cpp`:

```cpp
#include <coooda_cpp/ops/vector.hpp>
#include <coooda_core/status.hpp>

namespace coooda_cpp::ops {

std::vector<float> vector_add_reference(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        coooda_core::fail("vector_add_reference requires equal input sizes");
    }

    std::vector<float> out(a.size(), 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

} // namespace coooda_cpp::ops
```

- [ ] **Step 2: Add CUDA package**

`packages/coooda_cuda/include/coooda_cuda/package.cuh`:

```cpp
#pragma once

#include <string>

namespace coooda_cuda {

std::string backend_name();

} // namespace coooda_cuda
```

`packages/coooda_cuda/src/package.cu`:

```cpp
#include <coooda_cuda/package.cuh>

namespace coooda_cuda {

std::string backend_name() {
    return "coooda_cuda";
}

} // namespace coooda_cuda
```

`packages/coooda_cuda/include/coooda_cuda/ops/vector.cuh`:

```cpp
#pragma once

#include <vector>

namespace coooda_cuda::ops {

std::vector<float> vector_add_baseline(const std::vector<float> &a, const std::vector<float> &b);

} // namespace coooda_cuda::ops
```

`packages/coooda_cuda/src/ops/vector.cu`:

```cpp
#include <coooda_cuda/ops/vector.cuh>
#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>

#include <cuda_runtime.h>

namespace {

__global__ void vector_add_kernel(const float *a, const float *b, float *out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

} // namespace

namespace coooda_cuda::ops {

std::vector<float> vector_add_baseline(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        coooda_core::fail("vector_add_baseline requires equal input sizes");
    }

    if (a.empty()) {
        return {};
    }

    const std::size_t bytes = a.size() * sizeof(float);
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_out = nullptr;

    COODA_CUDA_CHECK(cudaMalloc(&device_a, bytes));
    COODA_CUDA_CHECK(cudaMalloc(&device_b, bytes));
    COODA_CUDA_CHECK(cudaMalloc(&device_out, bytes));
    COODA_CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
    COODA_CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int n = static_cast<int>(a.size());
    const int grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(device_a, device_b, device_out, n);
    COODA_CUDA_CHECK_LAST("vector_add_kernel");

    std::vector<float> out(a.size(), 0.0f);
    COODA_CUDA_CHECK(cudaMemcpy(out.data(), device_out, bytes, cudaMemcpyDeviceToHost));
    COODA_CUDA_CHECK(cudaFree(device_a));
    COODA_CUDA_CHECK(cudaFree(device_b));
    COODA_CUDA_CHECK(cudaFree(device_out));
    return out;
}

} // namespace coooda_cuda::ops
```

- [ ] **Step 3: Add compare package**

`packages/coooda_compare/include/coooda_compare/package.hpp`:

```cpp
#pragma once

#include <string>

namespace coooda_compare {

std::string package_name();

} // namespace coooda_compare
```

`packages/coooda_compare/src/package.cpp`:

```cpp
#include <coooda_compare/package.hpp>

namespace coooda_compare {

std::string package_name() {
    return "coooda_compare";
}

} // namespace coooda_compare
```

`packages/coooda_compare/include/coooda_compare/ops/vector_compare.hpp`:

```cpp
#pragma once

namespace coooda_compare::ops {

bool vector_add_matches_reference();

} // namespace coooda_compare::ops
```

`packages/coooda_compare/src/ops/vector_compare.cu`:

```cpp
#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_cpp/ops/vector.hpp>
#include <coooda_cuda/ops/vector.cuh>

#include <cmath>
#include <vector>

namespace coooda_compare::ops {

bool vector_add_matches_reference() {
    const std::vector<float> a{1.0f, -2.0f, 3.5f, 4.0f};
    const std::vector<float> b{0.5f, 2.0f, -1.5f, 8.0f};
    const std::vector<float> expected = coooda_cpp::ops::vector_add_reference(a, b);
    const std::vector<float> actual = coooda_cuda::ops::vector_add_baseline(a, b);

    if (expected.size() != actual.size()) {
        return false;
    }

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > 1.0e-6f) {
            return false;
        }
    }
    return true;
}

} // namespace coooda_compare::ops
```

- [ ] **Step 4: Configure and build package libraries**

Run:

```bash
cmake --preset debug
cmake --build --preset debug --target coooda_core coooda_cpp coooda_cuda coooda_compare
```

Expected: all four libraries build.

- [ ] **Step 5: Commit package stubs**

Run:

```bash
git add packages/coooda_cpp packages/coooda_cuda packages/coooda_compare
git commit -m "feat: add separated backend package stubs"
```

Expected: commit contains C++ backend, CUDA backend, and compare package files.

---

### Task 5: Add Tests Benchmarks And Apps

**Files:**
- Create: all `tests/`, `bench/`, and `apps/` files listed in File Structure.

- [ ] **Step 1: Add `tests/CMakeLists.txt`**

Write exactly:

```cmake
add_executable(coooda_cpp_tests cpp/test_cpp_smoke.cpp)
target_link_libraries(coooda_cpp_tests PRIVATE coooda_core coooda_cpp)
add_test(NAME coooda_cpp_tests COMMAND coooda_cpp_tests)

add_executable(coooda_cuda_tests cuda/test_cuda_smoke.cu)
target_link_libraries(coooda_cuda_tests PRIVATE coooda_core coooda_cuda)
add_test(NAME coooda_cuda_tests COMMAND coooda_cuda_tests)

add_executable(coooda_compare_tests compare/test_compare_smoke.cu)
target_link_libraries(coooda_compare_tests PRIVATE coooda_core coooda_compare)
add_test(NAME coooda_compare_tests COMMAND coooda_compare_tests)
```

- [ ] **Step 2: Add smoke tests**

`tests/cpp/test_cpp_smoke.cpp`:

```cpp
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/vector.hpp>
#include <coooda_cpp/package.hpp>

#include <vector>

int main() {
    return coooda_core::test::run_tests({
        {"cpp_backend_name", []() {
             coooda_core::test::require_equal("coooda_cpp", coooda_cpp::backend_name(), "backend name");
         }},
        {"cpp_vector_add_reference", []() {
             const std::vector<float> actual =
                 coooda_cpp::ops::vector_add_reference({1.0f, 2.0f}, {3.0f, 4.0f});
             coooda_core::test::require(actual.size() == 2, "vector add size");
             coooda_core::test::require(actual[0] == 4.0f, "vector add element 0");
             coooda_core::test::require(actual[1] == 6.0f, "vector add element 1");
         }},
    });
}
```

`tests/cuda/test_cuda_smoke.cu`:

```cpp
#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/ops/vector.cuh>
#include <coooda_cuda/package.cuh>

#include <vector>

int main() {
    return coooda_core::test::run_tests({
        {"cuda_backend_name", []() {
             coooda_core::test::require_equal("coooda_cuda", coooda_cuda::backend_name(), "backend name");
         }},
        {"cuda_vector_add_baseline", []() {
             const std::vector<float> actual =
                 coooda_cuda::ops::vector_add_baseline({1.0f, 2.0f}, {3.0f, 4.0f});
             coooda_core::test::require(actual.size() == 2, "vector add size");
             coooda_core::test::require(actual[0] == 4.0f, "vector add element 0");
             coooda_core::test::require(actual[1] == 6.0f, "vector add element 1");
         }},
    });
}
```

`tests/compare/test_compare_smoke.cu`:

```cpp
#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_compare/package.hpp>
#include <coooda_core/test_harness.hpp>

int main() {
    return coooda_core::test::run_tests({
        {"compare_package_name", []() {
             coooda_core::test::require_equal(
                 "coooda_compare", coooda_compare::package_name(), "package name"
             );
         }},
        {"compare_vector_add", []() {
             coooda_core::test::require(
                 coooda_compare::ops::vector_add_matches_reference(),
                 "CUDA vector add should match C++ reference"
             );
         }},
    });
}
```

- [ ] **Step 3: Add benchmark targets**

`bench/CMakeLists.txt`:

```cmake
add_executable(coooda_cpp_bench cpp/bench_cpp_smoke.cpp)
target_link_libraries(coooda_cpp_bench PRIVATE coooda_core coooda_cpp)

add_executable(coooda_cuda_bench cuda/bench_cuda_smoke.cu)
target_link_libraries(coooda_cuda_bench PRIVATE coooda_core coooda_cuda)

add_executable(coooda_compare_bench compare/bench_compare_smoke.cu)
target_link_libraries(coooda_compare_bench PRIVATE coooda_core coooda_compare)
```

`bench/cpp/bench_cpp_smoke.cpp`:

```cpp
#include <coooda_core/benchmark.hpp>
#include <coooda_cpp/ops/vector.hpp>

int main() {
    const auto result = coooda_core::bench::time_once("cpp_vector_add_reference", []() {
        (void)coooda_cpp::ops::vector_add_reference({1.0f, 2.0f}, {3.0f, 4.0f});
    });
    coooda_core::bench::print_result(result);
    return 0;
}
```

`bench/cuda/bench_cuda_smoke.cu`:

```cpp
#include <coooda_core/benchmark.hpp>
#include <coooda_cuda/ops/vector.cuh>

int main() {
    const auto result = coooda_core::bench::time_once("cuda_vector_add_baseline", []() {
        (void)coooda_cuda::ops::vector_add_baseline({1.0f, 2.0f}, {3.0f, 4.0f});
    });
    coooda_core::bench::print_result(result);
    return 0;
}
```

`bench/compare/bench_compare_smoke.cu`:

```cpp
#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_core/benchmark.hpp>

int main() {
    const auto result = coooda_core::bench::time_once("compare_vector_add", []() {
        (void)coooda_compare::ops::vector_add_matches_reference();
    });
    coooda_core::bench::print_result(result);
    return 0;
}
```

- [ ] **Step 4: Add app targets**

`apps/CMakeLists.txt`:

```cmake
add_executable(inspect_device inspect_device.cu)
target_link_libraries(inspect_device PRIVATE coooda_core)

add_executable(run_checkpoint run_checkpoint.cpp)
target_link_libraries(run_checkpoint PRIVATE coooda_core coooda_cpp)

add_executable(compare_backends compare_backends.cu)
target_link_libraries(compare_backends PRIVATE coooda_core coooda_compare)

add_executable(tiny_gpt_cpp tiny_gpt_cpp.cpp)
target_link_libraries(tiny_gpt_cpp PRIVATE coooda_core coooda_cpp)

add_executable(tiny_gpt_cuda tiny_gpt_cuda.cu)
target_link_libraries(tiny_gpt_cuda PRIVATE coooda_core coooda_cuda)
```

`apps/inspect_device.cu`:

```cpp
#include <coooda_core/device.hpp>

#include <iostream>

int main() {
    try {
        std::cout << coooda_core::cuda::device_summary() << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cout << "CUDA device inspection skipped: " << error.what() << "\n";
        return 0;
    }
}
```

`apps/run_checkpoint.cpp`:

```cpp
#include <coooda_cpp/ops/vector.hpp>

#include <iostream>

int main() {
    const auto out = coooda_cpp::ops::vector_add_reference({1.0f}, {2.0f});
    std::cout << "Checkpoint runner ready. cpp vector smoke result=" << out[0] << "\n";
    return 0;
}
```

`apps/compare_backends.cu`:

```cpp
#include <coooda_compare/ops/vector_compare.hpp>

#include <iostream>

int main() {
    const bool ok = coooda_compare::ops::vector_add_matches_reference();
    std::cout << "Backend comparison smoke: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
```

`apps/tiny_gpt_cpp.cpp`:

```cpp
#include <iostream>

int main() {
    std::cout << "tiny_gpt_cpp scaffold ready. Follow course/16_tiny_gpt.md to build this app.\n";
    return 0;
}
```

`apps/tiny_gpt_cuda.cu`:

```cpp
#include <iostream>

int main() {
    std::cout << "tiny_gpt_cuda scaffold ready. Follow course/16_tiny_gpt.md to build this app.\n";
    return 0;
}
```

- [ ] **Step 5: Build and run tests**

Run:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Expected: `coooda_cpp_tests`, `coooda_cuda_tests`, and `coooda_compare_tests` pass.

- [ ] **Step 6: Run smoke apps**

Run:

```bash
./build/debug/apps/inspect_device
./build/debug/apps/run_checkpoint
./build/debug/apps/compare_backends
./build/debug/apps/tiny_gpt_cpp
./build/debug/apps/tiny_gpt_cuda
```

Expected:

- `inspect_device` prints device info or a graceful skip message.
- `run_checkpoint` prints `cpp vector smoke result=3`.
- `compare_backends` prints `Backend comparison smoke: PASS`.
- Tiny GPT apps print scaffold-ready messages.

- [ ] **Step 7: Commit tests benchmarks and apps**

Run:

```bash
git add tests bench apps
git commit -m "feat: add scaffold tests benches and apps"
```

Expected: commit contains only tests, benchmarks, and app files.

---

### Task 6: Add CPU-Then-CUDA Course Files

**Files:**
- Create: all files under `course/` listed in File Structure.

- [ ] **Step 1: Create `course/00_start_here.md`**

Write a guide with these exact sections:

```markdown
# 00: Start Here

Coooda is built in tiny checkpoints. Each topic keeps the C++ and CUDA work together.

## Loop

1. Write the C++ reference.
2. Run the C++ test.
3. Write the CUDA baseline.
4. Run the CUDA test.
5. Run the compare test.
6. Optimize CUDA only after correctness passes.

## Commands

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

## Package Map

- `packages/coooda_cpp`: readable C++ reference backend.
- `packages/coooda_cuda`: CUDA backend.
- `packages/coooda_core`: shared infrastructure.
- `packages/coooda_compare`: backend equivalence checks.

## Rule

Do not add private one-off code to apps. Apps use library modules.
```

- [ ] **Step 2: Create `course/01_project_scaffold.md`**

Include these checkpoints:

```markdown
# 01: Project Scaffold

## Checkpoint 01.01: Configure Debug Build

Goal: prove CMake can configure the workspace.

You write: nothing.

Run:
- `cmake --preset debug`

Pass:
- Build files are generated in `build/debug`.

## Checkpoint 01.02: Build Empty Scaffold

Goal: prove all package, test, benchmark, and app targets compile.

You write: nothing.

Run:
- `cmake --build --preset debug`

Pass:
- The build reaches 100%.

## Checkpoint 01.03: Run Smoke Tests

Goal: prove the test harness is ready.

You write: nothing.

Run:
- `ctest --preset debug`

Pass:
- C++ smoke, CUDA smoke, and compare smoke tests pass.
```

- [ ] **Step 3: Create topic files `02_core_helpers.md` through `16_tiny_gpt.md`**

Create each topic file from the table after this checkpoint skeleton. Replace the title, checkpoint numbers, target paths, and benchmark command with the values from that same table before writing the file.

```markdown
# 04: Vector Ops

## Checkpoint 04.01: C++ Reference

Goal:
Implement the C++ reference functions for vector add, SAXPY, ReLU, sigmoid, and elementwise multiply.

You write:
- `packages/coooda_cpp/include/coooda_cpp/ops/vector.hpp`
- `packages/coooda_cpp/src/ops/vector.cpp`
- `tests/cpp/test_vector_ops.cpp`

Already provided:
- Test harness
- Mismatch reporting
- Build target wiring

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R vector_ops_cpp`

Pass:
- The C++ vector operation tests pass.

## Checkpoint 04.02: CUDA Baseline

Goal:
Implement matching one-thread-per-element CUDA baselines.

You write:
- `packages/coooda_cuda/include/coooda_cuda/ops/vector.cuh`
- `packages/coooda_cuda/src/ops/vector.cu`
- `tests/cuda/test_vector_ops.cu`

Already provided:
- CUDA error checks
- Device inspection
- Host comparison utilities

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R vector_ops_cuda`

Pass:
- CUDA vector operations match the expected small cases.

## Checkpoint 04.03: Compare Backends

Goal:
Prove CUDA matches the C++ reference.

You write:
- `packages/coooda_compare/include/coooda_compare/ops/vector_compare.hpp`
- `packages/coooda_compare/src/ops/vector_compare.cu`
- `tests/compare/test_vector_ops_compare.cu`

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R vector_ops_compare`

Pass:
- The compare test reports no mismatch for exact, boundary, and seeded random cases.

## Checkpoint 04.04: Grid-Stride CUDA Variant

Goal:
Add grid-stride CUDA variants and compare them against the baseline.

You write:
- `packages/coooda_cuda/include/coooda_cuda/ops/vector.cuh`
- `packages/coooda_cuda/src/ops/vector.cu`
- `bench/cuda/bench_vector_ops.cu`
- `bench/compare/bench_vector_ops_compare.cu`

Run:
- `cmake --build --preset release`
- `./build/release/bench/coooda_cuda_bench --case vector_ops`
- `./build/release/bench/coooda_compare_bench --case vector_ops`

Pass:
- Correctness still passes before any speed claim.
```

Use the same concrete structure for these topic files:

```text
02_core_helpers.md: title "02: Core Helpers"; C++ target paths packages/coooda_core/include/coooda_core/status.hpp, packages/coooda_core/include/coooda_core/tensor.hpp, packages/coooda_core/src/status.cpp, packages/coooda_core/src/tensor.cpp, tests/cpp/test_core_helpers.cpp; CUDA target paths packages/coooda_core/include/coooda_core/cuda_check.cuh, packages/coooda_core/include/coooda_core/device.hpp, packages/coooda_core/src/cuda_check.cu, packages/coooda_core/src/device.cu, tests/cuda/test_core_cuda.cu; compare paths tests/compare/test_core_compare.cu; release command ./build/release/bench/coooda_compare_bench --case core_helpers.
03_memory_and_tensors.md: title "03: Memory And Tensors"; C++ target paths packages/coooda_core/include/coooda_core/host_buffer.hpp, packages/coooda_core/include/coooda_core/tensor.hpp, packages/coooda_core/src/tensor.cpp, tests/cpp/test_memory_tensors.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/memory/device_buffer.cuh, packages/coooda_cuda/src/memory/device_buffer.cu, tests/cuda/test_memory_tensors.cu; compare paths tests/compare/test_memory_tensors_compare.cu; release command ./build/release/bench/coooda_compare_bench --case memory_tensors.
04_vector_ops.md: title "04: Vector Ops"; C++ target paths packages/coooda_cpp/include/coooda_cpp/ops/vector.hpp, packages/coooda_cpp/src/ops/vector.cpp, tests/cpp/test_vector_ops.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/ops/vector.cuh, packages/coooda_cuda/src/ops/vector.cu, tests/cuda/test_vector_ops.cu; compare paths packages/coooda_compare/include/coooda_compare/ops/vector_compare.hpp, packages/coooda_compare/src/ops/vector_compare.cu, tests/compare/test_vector_ops_compare.cu; release command ./build/release/bench/coooda_compare_bench --case vector_ops.
05_reductions.md: title "05: Reductions"; C++ target paths packages/coooda_cpp/include/coooda_cpp/ops/reductions.hpp, packages/coooda_cpp/src/ops/reductions.cpp, tests/cpp/test_reductions.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/ops/reductions.cuh, packages/coooda_cuda/src/ops/reductions.cu, tests/cuda/test_reductions.cu; compare paths tests/compare/test_reductions_compare.cu; release command ./build/release/bench/coooda_compare_bench --case reductions.
06_matmul.md: title "06: Matmul"; C++ target paths packages/coooda_cpp/include/coooda_cpp/ops/matmul.hpp, packages/coooda_cpp/src/ops/matmul.cpp, tests/cpp/test_matmul.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/ops/matmul.cuh, packages/coooda_cuda/src/ops/matmul.cu, tests/cuda/test_matmul.cu; compare paths tests/compare/test_matmul_compare.cu; release command ./build/release/bench/coooda_compare_bench --case matmul.
07_elementwise_and_fusion.md: title "07: Elementwise And Fusion"; C++ target paths packages/coooda_cpp/include/coooda_cpp/ops/elementwise.hpp, packages/coooda_cpp/src/ops/elementwise.cpp, tests/cpp/test_elementwise.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/ops/elementwise.cuh, packages/coooda_cuda/src/ops/elementwise.cu, tests/cuda/test_elementwise.cu; compare paths tests/compare/test_elementwise_compare.cu; release command ./build/release/bench/coooda_compare_bench --case elementwise.
08_softmax_and_loss.md: title "08: Softmax And Loss"; C++ target paths packages/coooda_cpp/include/coooda_cpp/nn/loss.hpp, packages/coooda_cpp/src/nn/loss.cpp, tests/cpp/test_softmax_loss.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/nn/loss.cuh, packages/coooda_cuda/src/nn/loss.cu, tests/cuda/test_softmax_loss.cu; compare paths tests/compare/test_softmax_loss_compare.cu; release command ./build/release/bench/coooda_compare_bench --case softmax_loss.
09_normalization.md: title "09: Normalization"; C++ target paths packages/coooda_cpp/include/coooda_cpp/nn/normalization.hpp, packages/coooda_cpp/src/nn/normalization.cpp, tests/cpp/test_normalization.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/nn/normalization.cuh, packages/coooda_cuda/src/nn/normalization.cu, tests/cuda/test_normalization.cu; compare paths tests/compare/test_normalization_compare.cu; release command ./build/release/bench/coooda_compare_bench --case normalization.
10_embeddings.md: title "10: Embeddings"; C++ target paths packages/coooda_cpp/include/coooda_cpp/nn/embedding.hpp, packages/coooda_cpp/src/nn/embedding.cpp, tests/cpp/test_embeddings.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/nn/embedding.cuh, packages/coooda_cuda/src/nn/embedding.cu, tests/cuda/test_embeddings.cu; compare paths tests/compare/test_embeddings_compare.cu; release command ./build/release/bench/coooda_compare_bench --case embeddings.
11_linear_and_mlp.md: title "11: Linear And MLP"; C++ target paths packages/coooda_cpp/include/coooda_cpp/nn/linear.hpp, packages/coooda_cpp/include/coooda_cpp/nn/mlp.hpp, packages/coooda_cpp/src/nn/linear.cpp, packages/coooda_cpp/src/nn/mlp.cpp, tests/cpp/test_linear_mlp.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/nn/linear.cuh, packages/coooda_cuda/include/coooda_cuda/nn/mlp.cuh, packages/coooda_cuda/src/nn/linear.cu, packages/coooda_cuda/src/nn/mlp.cu, tests/cuda/test_linear_mlp.cu; compare paths tests/compare/test_linear_mlp_compare.cu; release command ./build/release/bench/coooda_compare_bench --case linear_mlp.
12_backward_and_optimizers.md: title "12: Backward And Optimizers"; C++ target paths packages/coooda_cpp/include/coooda_cpp/nn/backward.hpp, packages/coooda_cpp/include/coooda_cpp/optim/sgd.hpp, packages/coooda_cpp/src/nn/backward.cpp, packages/coooda_cpp/src/optim/sgd.cpp, tests/cpp/test_backward_optimizers.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/nn/backward.cuh, packages/coooda_cuda/include/coooda_cuda/optim/sgd.cuh, packages/coooda_cuda/src/nn/backward.cu, packages/coooda_cuda/src/optim/sgd.cu, tests/cuda/test_backward_optimizers.cu; compare paths tests/compare/test_backward_optimizers_compare.cu; release command ./build/release/bench/coooda_compare_bench --case backward_optimizers.
13_tiny_mlp_training.md: title "13: Tiny MLP Training"; C++ target paths packages/coooda_cpp/include/coooda_cpp/training/tiny_mlp.hpp, packages/coooda_cpp/src/training/tiny_mlp.cpp, tests/cpp/test_tiny_mlp_training.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/training/tiny_mlp.cuh, packages/coooda_cuda/src/training/tiny_mlp.cu, tests/cuda/test_tiny_mlp_training.cu; compare paths tests/compare/test_tiny_mlp_training_compare.cu; release command ./build/release/bench/coooda_compare_bench --case tiny_mlp_training.
14_attention.md: title "14: Attention"; C++ target paths packages/coooda_cpp/include/coooda_cpp/nn/attention.hpp, packages/coooda_cpp/src/nn/attention.cpp, tests/cpp/test_attention.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/nn/attention.cuh, packages/coooda_cuda/src/nn/attention.cu, tests/cuda/test_attention.cu; compare paths tests/compare/test_attention_compare.cu; release command ./build/release/bench/coooda_compare_bench --case attention.
15_transformer_block.md: title "15: Transformer Block"; C++ target paths packages/coooda_cpp/include/coooda_cpp/nn/transformer_block.hpp, packages/coooda_cpp/src/nn/transformer_block.cpp, tests/cpp/test_transformer_block.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/nn/transformer_block.cuh, packages/coooda_cuda/src/nn/transformer_block.cu, tests/cuda/test_transformer_block.cu; compare paths tests/compare/test_transformer_block_compare.cu; release command ./build/release/bench/coooda_compare_bench --case transformer_block.
16_tiny_gpt.md: title "16: Tiny GPT"; C++ target paths packages/coooda_cpp/include/coooda_cpp/models/tiny_gpt.hpp, packages/coooda_cpp/src/models/tiny_gpt.cpp, apps/tiny_gpt_cpp.cpp, tests/cpp/test_tiny_gpt.cpp; CUDA target paths packages/coooda_cuda/include/coooda_cuda/models/tiny_gpt.cuh, packages/coooda_cuda/src/models/tiny_gpt.cu, apps/tiny_gpt_cuda.cu, tests/cuda/test_tiny_gpt.cu; compare paths tests/compare/test_tiny_gpt_compare.cu; release command ./build/release/bench/coooda_compare_bench --case tiny_gpt.
```

- [ ] **Step 4: Commit course files**

Run:

```bash
git add course
git commit -m "docs: add cpu then cuda checkpoint course"
```

Expected: commit contains only `course/`.

---

### Task 7: Final Verification And Cleanup

**Files:**
- Modify only files needed to fix build or test failures from prior tasks.

- [ ] **Step 1: Run full debug verification**

Run:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Expected: configure succeeds, build succeeds, all tests pass.

- [ ] **Step 2: Run release build**

Run:

```bash
cmake --preset release
cmake --build --preset release
```

Expected: release build succeeds.

- [ ] **Step 3: Run apps**

Run:

```bash
./build/debug/apps/inspect_device
./build/debug/apps/run_checkpoint
./build/debug/apps/compare_backends
```

Expected: `inspect_device` exits 0, `run_checkpoint` exits 0, and `compare_backends` exits 0.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: only intentional files are changed or committed. No generated `build/` files are tracked.

- [ ] **Step 5: Final commit if any verification fixes were needed**

If Step 1 through Step 4 required fixes, run:

```bash
git add CMakeLists.txt CMakePresets.json README.md .gitignore packages tests bench apps course
git commit -m "fix: stabilize scaffold verification"
```

Expected: final commit contains only verification fixes.

---

## Self-Review

- Spec coverage: This plan covers the package split, CMake scaffold, boilerplate helpers, test harness, benchmark harness, app runners, one-file CPU-plus-CUDA course units, and final verification.
- Placeholder scan: No task uses TBD, TODO, or implementation-later placeholders as required work. Course files deliberately describe future user checkpoints, while the starter scaffold itself remains runnable.
- Type consistency: Package names are consistent across CMake, headers, source files, tests, benchmarks, and apps: `coooda_core`, `coooda_cpp`, `coooda_cuda`, and `coooda_compare`.
