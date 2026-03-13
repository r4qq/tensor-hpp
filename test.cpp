#include <cstdint>
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <chrono>
#include "Tensor.hpp"

void testConstruction()
{
    Tensor::Tensor<int> t({2, 3});
    assert(t.rank() == 2);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
    assert(t.size() == 6);
}

void testElementAccess()
{
    Tensor::Tensor<int> t({2, 2});
    t(0, 0) = 1;
    t(0, 1) = 2;
    t(1, 0) = 3;
    t(1, 1) = 4;

    assert(t(0, 0) == 1);
    assert(t(1, 1) == 4);
}

void testOutOfBounds()
{
    Tensor::Tensor<int> t({2, 2});
    bool caught = false;

    try {
        t(3, 0);
    } catch (const std::out_of_range&) {
        caught = true;
    }

    assert(caught);
}

void testElementWiseOps()
{
    Tensor::Tensor<int> a({2, 2});
    Tensor::Tensor<int> b({2, 2});

    a.fill(2);
    b.fill(3);

    auto c = a + b;
    assert(c(0,0) == 5);

    c = b - a;
    assert(c(0,0) == 1);

    c = a * b;
    assert(c(0,0) == 6);
}

void testScalarOps()
{
    Tensor::Tensor<int> t({2, 2});
    t.fill(4);

    auto result = t * 2;
    assert(result(0,0) == 8);

    result /= 2;
    assert(result(0,0) == 4);
}

void testMatmul()
{
    Tensor::Tensor<int> a({2, 3});
    Tensor::Tensor<int> b({3, 2});
    Tensor::Tensor<int> c({2, 2});

    int val = 1;
    for (uint64_t i = 0; i < 2; ++i)
        for (uint64_t j = 0; j < 3; ++j)
            a(i, j) = val++;

    val = 1;
    for (uint64_t i = 0; i < 3; ++i)
        for (uint64_t j = 0; j < 2; ++j)
            b(i, j) = val++;

    Tensor::matmul(a, b, c);

    assert(c(0,0) == 22);
    assert(c(0,1) == 28);
    assert(c(1,0) == 49);
    assert(c(1,1) == 64);
}

void testTranspose()
{
    Tensor::Tensor<int> a({2, 3});
    Tensor::Tensor<int> b({3, 2});

    int val = 1;
    for (uint64_t i = 0; i < 2; ++i)
        for (uint64_t j = 0; j < 3; ++j)
            a(i, j) = val++;

    Tensor::transpose(a, b);

    assert(b(0,0) == 1);
    assert(b(1,0) == 2);
    assert(b(2,0) == 3);
    assert(b(0,1) == 4);
    assert(b(1,1) == 5);
    assert(b(2,1) == 6);
}

void testEquality()
{
    Tensor::Tensor<int> a({2,2});
    Tensor::Tensor<int> b({2,2});

    a.fill(5);
    b.fill(5);

    assert(a == b);

    b(0,0) = 3;
    assert(a != b);
}

void testMatvec()
{
    // Matrix (2x3)
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    Tensor::Tensor<int> a({2, 3});
    
    // Vector (3)
    // [ 7, 8, 9 ]
    Tensor::Tensor<int> x({3});
    
    // Output Vector (2)
    Tensor::Tensor<int> y({2});

    // Initialize Matrix
    int val = 1;
    for (uint64_t i = 0; i < 2; ++i)
        for (uint64_t j = 0; j < 3; ++j)
            a(i, j) = val++;

    // Initialize Input Vector
    x(0) = 7; x(1) = 8; x(2) = 9;

    Tensor::matvec(a, x, y);

    // Expected:
    // row0: (1*7) + (2*8) + (3*9) = 7 + 16 + 27 = 50
    // row1: (4*7) + (5*8) + (6*9) = 28 + 40 + 54 = 122
    assert(y(0) == 50);
    assert(y(1) == 122);
}

void benchmarkMatmul()
{
    using Clock = std::chrono::high_resolution_clock;

    const uint64_t N = 1000;   // adjust if needed
    const int testRuns = 20;
    const int warmupRuns = 50; 

    std::cout << "\nBenchmarking matmul with "
              << N << "x" << N << " matrices\n";

    Tensor::Tensor<double> a({N, N});
    Tensor::Tensor<double> b({N, N});
    Tensor::Tensor<double> c({N, N});

    for (uint64_t i = 0; i < N; ++i)
        for (uint64_t j = 0; j < N; ++j)
        {
            a(i, j) = static_cast<double>((i + j) % 10);
            b(i, j) = static_cast<double>((i * j) % 10);
        }

    // Warm-up run
    std::cout << "Warming up CPU (Burning PL2 state) with " << warmupRuns << " runs" << std::endl;

    // 1. Hammer the CPU to build heat and trigger the AVX offset
    for (int w = 0; w < warmupRuns; ++w) 
    {
        Tensor::matmul(a, b, c);
    }

    // 2. The Compiler Sink
    // Force the compiler to evaluate the math by reading a single value.
    // 'volatile' ensures the compiler cannot optimize this read away.
    volatile float sink = c.unchecked(0, 0);

    std::cout << "Warm-up complete. Starting benchmark..." << std::endl;


    double totalMs = 0.0;

    for (int r = 0; r < testRuns; ++r)
    {
        auto start = Clock::now();

        Tensor::matmul(a, b, c);

        auto end = Clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        totalMs += elapsed.count();

        // prevent optimization removal
        volatile double sink = c(0,0);
        (void)sink;

        std::cout << "Run " << r + 1 << ": "
                  << elapsed.count() << " ms\n";

        double flops = 2.0 * N * N * N;
        double gflops = (flops / 1e9) / (elapsed.count() / 1000.0);

        std::cout << "GFLOPS: " << gflops << "\n";
    }

    std::cout << "Average: " << (totalMs / testRuns)
              << " ms\n\n";
}

void benchmarkMatvec()
{
    using Clock = std::chrono::high_resolution_clock;

    const uint64_t N = 1000;
    const int testRuns = 20;
    const int warmupRuns = 50;

    std::cout << "Benchmarking matvec with " 
              << N << "x" << N << " matrix\n"
              << "and " << N << " size vector\n";

    Tensor::Tensor<double> a({N, N});
    Tensor::Tensor<double> x({N});
    Tensor::Tensor<double> y({N});

    for(uint64_t i = 0; i < N; ++i)
    {
        x(i) = static_cast<double>((i) % 10);
        for(uint64_t j = 0; j < N; ++j)
        {
            a(i, j) = static_cast<double>((i + j) % 10);
        }
    }

    std::cout << "Warming up CPU (Burning PL2 state) with " << warmupRuns << " runs" << std::endl;

    for (uint64_t w = 0; w < warmupRuns; w++) 
    {
        Tensor::matvec(a, x, y);
    }

    volatile double sink = y(0);

    std::cout << "Warm-up complete. Starting benchmark..." << std::endl;

    double totalMs = 0.0; 

    for (uint64_t r = 0; r < testRuns; ++r) 
    {
        auto start = Clock::now();

        Tensor::matvec(a, x, y);

        auto stop = Clock::now();

        std::chrono::duration<double, std::milli> elapsed = stop - start;
        totalMs += elapsed.count();

        volatile double sink = y(0);

        (void)sink;

        std::cout << "Run " << r + 1 << ": "
                  << elapsed.count() << " ms\n";

        double flops = 2.0 * N * N;
        double gflops = (flops / 1e9) / (elapsed.count() / 1000.0);

        std::cout << "GFLOPS: " << gflops << "\n";
    }

    std::cout << "Average: " << (totalMs / testRuns)
              << " ms\n\n";
}

int main()
{
    testConstruction();
    testElementAccess();
    testOutOfBounds();
    testElementWiseOps();
    testScalarOps();
    testMatmul();
    testTranspose();
    testEquality();
    testMatvec();

    std::cout << "All correctness tests passed.\n";

    benchmarkMatmul();
    benchmarkMatvec();

    return 0;
}