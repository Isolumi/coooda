# Milestone 00: Project Setup And CMake

## Deliverable

An empty CUDA/C++ project named `coooda` with build, test, benchmark, and profile entry points.

## Implement

- Target `coooda`
- Target `coooda_tests`
- Target `coooda_bench`
- Target `coooda_tiny_gpt`
- Directory `include/coooda`
- Directory `src`
- Directory `tests`
- Directory `bench`
- Directory `apps`
- Directory `notes`

## Steps

1. Create the project tree.
2. Create the root CMake build.
3. Enable C++ and CUDA.
4. Add the `coooda` library target.
5. Add an empty test executable and register it with CTest.
6. Add an empty benchmark executable that is run manually.
7. Add debug, release, and profile configure commands to `notes/build.md`.

## Done

- Debug configure works.
- Release configure works.
- `coooda_tests` runs through CTest.
- `coooda_bench` runs manually.
- No real kernels exist yet.
