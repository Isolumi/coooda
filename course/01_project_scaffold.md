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
- `ctest --preset debug --no-tests=error`

Pass:
- C++ smoke, CUDA smoke, and compare smoke tests pass.
