## tensor-hpp

#### Note: Educational Project
A lightweight, header-only C++17 library built to explore hardware-level optimizations, template metaprogramming, and linear algebra performance. Not intended for production.

### Architecture & Optimizations

 - Modern-ish C++: Variadic templates for N-dimensional indexing, if constexpr branching, and compile-time type constraints (std::is_arithmetic).

 - Hardware Intrinsics: Explicit SIMD vectorization utilizing AVX2 and Fused Multiply-Add (FMA).

 - Memory Hierarchy: Row-major contiguous layout, explicit loop unrolling, and L1/L2 cache-tiled blocking to minimize cache misses.

 - Multithreading: OpenMP  for parallel scaling across CPU cores in every implementation.

### Implementations

The library exposes three headers to show optimization progression:

 - Tensor.hpp: Standard C++ STL implementation. Focuses on correctness and generic N-dimensional memory layout.

 - Tensor-simd.hpp: Replaces standard loops with AVX2/FMA intrinsic kernels (8-wide float, 4-wide double) and loop unrolling.

 - Tensor-simd-block.hpp: Combines AVX2 kernels with L1 cache blocking (tiled matrix multiplication).

 - Tensor-simd-block-multi.hpp: Adds OpenMP multithreading.


### Requirements & Compilation

 - C++17 

 - CPU with AVX2 and FMA support

 - OpenMP (Required for multi-core) 

### Example

```cpp
#include "Tensor-simd-block.hpp"

int main() {
    Tensor::Tensor<float> A({1000, 1000});
    Tensor::Tensor<float> B({1000, 1000});
    Tensor::Tensor<float> C({1000, 1000});

    A.fill(1.5f);
    B.fill(2.0f);

    A += B; 

    Tensor::matmul(A, B, C);

    Tensor::Tensor<float> x({1000});
    Tensor::Tensor<float> y({1000});
    x.fill(1.0f);

    Tensor::matvec(A, x, y);

    return 0;
}
```

### Error Handling

 - std::invalid_argument: Shape mismatch, incorrect index count, or division by zero.

 - std::out_of_range: Out-of-bounds memory access.

 - std::runtime_error: Dimensionality requirements not met (e.g., non-2D matmul).

### License

MIT License. See the LICENSE file for details.

#### Author: r4qq (2025-2026)