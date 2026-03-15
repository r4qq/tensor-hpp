# tensor-hpp

A lightweight, header-only C++ library implementing generic N-dimensional tensors with optional SIMD and cache-blocked optimizations.

## Key Features

- Generic: Supports any arithmetic type (int, float, double, etc.).
- N-Dimensional: Create tensors of any rank and shape.
- Performance Optimized:
    * Standard: Clean, STL-based implementation for general use.
    * SIMD: AVX2 and FMA accelerated kernels for float and double operations.
    * Cache-Blocked: Tiled matrix operations (BLOCK_SIZE 32) to maximize L1/L2 cache efficiency for large datasets.
- Element Access:
    * Safe (bounds-checked) via the () operator.
    * Fast (unchecked) via the unchecked() method.
- Arithmetic: Full support for element-wise operations (+, -, *, /) and in-place operators (+=, -=, etc.).
- Linear Algebra:
    * Matrix multiplication (matmul) for 2D tensors.
    * Matrix-vector multiplication (matvec).
    * Transposition for 2D tensors.
- Safety: Throws exceptions for shape mismatches or out-of-bounds access.

## Requirements

- Compiler supporting C++17 or later.
- AVX2 Support: Required if using Tensor-simd.hpp or Tensor-simd-block.hpp.
- No external libraries (relies only on the STL and <immintrin.h> for SIMD).

## Installation

Since this is a header-only library, installation is as simple as copying the header file into your project.

Options avaiable:

  - Tensor.hpp: Standard version.
  - Tensor-simd.hpp: AVX2 optimized.
  - Tensor-simd-block.hpp: AVX2 + Cache Blocking (Recommended for large matrices).


## Usage

### 1. Creating a Tensor
Tensors are initialized with a vector specifying their dimensions (shape).

Example:

```cpp
    #include "Tensor-simd-block.hpp"

    Tensor::Tensor<float> A({1000, 1000});
    A.fill(1.0f);
```

### 2. Matrix and Vector Operations
Specific optimized routines are available for rank-2 tensors (matrices) and rank-1 tensors (vectors).

Example:

```cpp
    #include "Tensor-simd.hpp"

    Tensor::Tensor<double> MatA({1000, 1000});
    Tensor::Tensor<double> MatB({1000, 1000});
    Tensor::Tensor<double> Result({1000, 1000});

    Tensor::matmul(MatA, MatB, Result);

    Tensor::Tensor<double> x({1000});
    Tensor::Tensor<double> y({1000});

    Tensor::matvec(MatA, x, y);
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