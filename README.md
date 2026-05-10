# tensor-hpp

**Note: Educational Project**
This is a lightweight, header-only C++17 library developed to explore hardware-level optimizations, modern C++ template metaprogramming, and linear algebra implementation. It is not intended for production environments.

## Technical Overview

The project implements a generic N-dimensional tensor class with an emphasis on performance scaling. It demonstrates the progression from a standard C++ implementation to a highly optimized kernel utilizing SIMD instructions and CPU cache awareness.

### Core Concepts Applied
* **Modern C++:** Variadic templates for N-dimensional indexing, `if constexpr` for compile-time branch evaluation, and compile-time type assertions (`std::is_arithmetic`).
* **Hardware Optimization:** Explicit SIMD vectorization using AVX2 and Fused Multiply-Add (FMA) intrinsics.
* **Memory Architecture:** Row-major contiguous memory layout, loop unrolling, and cache-tiled matrix multiplication (blocking) to minimize L1/L2 cache misses.
* **API Design:** Operator overloading for element-wise arithmetic, providing both strict bounds-checked access via `()` and fast unchecked access for hot loops.

## Repository Structure

The library is separated into three distinct implementations to demonstrate the impact of different optimization techniques:

* **`Tensor.hpp`**
  A standard, STL-compliant implementation. Focuses on correctness, memory layout, and generic N-dimensional logic.
* **`Tensor-simd.hpp`**
  Replaces standard loops in `matmul` and `matvec` with AVX2/FMA intrinsic kernels (8-wide for `float`, 4-wide for `double`). Includes loop unrolling to maximize register usage.
* **`Tensor-simd-block.hpp`**
  Combines the AVX2 kernels with cache blocking (BLOCK_SIZE 32). Partitions matrices into smaller tiles to ensure data remains resident in the CPU cache during multi-pass computations.

## System Requirements

* C++17 compliant compiler.
* CPU with AVX2 and FMA support (for SIMD headers).
* Zero third-party dependencies (utilizes STL and `<immintrin.h>`).

## Example

```cpp
#include "Tensor-simd-block.hpp"

int main() {
    // Initialize 2D tensors (matrices)
    Tensor::Tensor<float> A({1000, 1000});
    Tensor::Tensor<float> B({1000, 1000});
    Tensor::Tensor<float> C({1000, 1000});

    A.fill(1.5f);
    B.fill(2.0f);

    // Element-wise arithmetic operations
    A += B;

    // Cache-blocked, SIMD-accelerated matrix multiplication
    Tensor::matmul(A, B, C);

    // Matrix-vector multiplication
    Tensor::Tensor<float> x({1000});
    Tensor::Tensor<float> y({1000});
    x.fill(1.0f);

    Tensor::matvec(A, x, y);

    return 0;
}
```

## Error Handling
The library throws standard C++ exceptions:
- std::invalid_argument: Shape mismatch, incorrect index count, or division by zero.
- std::out_of_range: Out-of-bounds access.
- std::runtime_error: Matrix operations on non-2D tensors or result shape mismatches.

## License
Project released under the MIT License. See the LICENSE file for details.

---
Author: r4qq (2025-2026)