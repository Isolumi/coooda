# Testing And Validation

Every operation follows the same pattern:

1. CPU reference.
2. Small exact test.
3. Boundary-shape tests.
4. Random stress test with recorded seed.
5. CUDA baseline comparison.
6. Optimized CUDA comparison.

## Failure Report Fields

Report:

- operation
- shape
- seed
- first mismatch
- expected value
- actual value
- absolute difference
- relative difference
- tolerance
- maximum observed error

## Gradient Checks

Use finite differences only on tiny shapes. The goal is confidence, not speed.
