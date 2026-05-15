# Build Notes

Run these commands from the course repository root.

## Configure

```bash
cmake -S coooda -B coooda/build/debug -DCMAKE_BUILD_TYPE=Debug
cmake -S coooda -B coooda/build/release -DCMAKE_BUILD_TYPE=Release
cmake -S coooda -B coooda/build/profile -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

The default CUDA architecture is `native`. Override it when needed:

```bash
cmake -S coooda -B coooda/build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
```

## Build

```bash
cmake --build coooda/build/debug -j
cmake --build coooda/build/release -j
cmake --build coooda/build/profile -j
```

## Test

```bash
ctest --test-dir coooda/build/debug --output-on-failure
```

## Benchmark

```bash
./coooda/build/release/coooda_bench
```

## Profile

```bash
nsys profile -o coooda/build/profile/coooda_bench_nsys ./coooda/build/profile/coooda_bench
ncu --set full --target-processes all ./coooda/build/profile/coooda_bench
```

## Clean

```bash
rm -rf coooda/build/debug coooda/build/release coooda/build/profile
```
