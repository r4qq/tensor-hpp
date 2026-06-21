/**
 * @file Tensor.hpp
 * @brief Lightweight generic N-dimensional tensor with SIMD support.
 * @author r4qq 
 * @date 2025-2026
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <sys/types.h>
#include <type_traits>
#include <utility>
#include <vector>
#include <immintrin.h>

#define BLOCK_SIZE 256

namespace Tensor
{
    /// N-dimensional generic tensor for numeric types.
    template<typename T>
    class Tensor
    {
        static_assert(std::is_arithmetic<T>::value, "Type must be numeric");

    private:
        std::vector<uint64_t> _shape;     ///< Tensor dimensions.
        std::vector<uint64_t> _strides;   ///< Strides for flat indexing.
        std::vector<T> _data;             ///< Contiguous data storage.

        /// Applies an element-wise binary operation.
        template<typename BinaryOp>
        Tensor<T> elementWiseOp(const Tensor<T>& otherTensor, BinaryOp op) const
        {
            if (_shape != otherTensor._shape)
            {
                throw std::invalid_argument("Tensor shape mismatch");
            }

            Tensor<T> result(_shape);
            std::transform(_data.begin(),
                           _data.end(),
                           otherTensor._data.begin(),
                           result._data.begin(),
                           op);
            return result;
        }

        /// Computes 1D index from N-dimensional indices.
        template<std::uint64_t... I, typename... Indices>
        inline uint64_t computeFlatUnrolled(std::index_sequence<I...>, Indices... idxs) const
        {
            return ((static_cast<uint64_t>(idxs) * _strides[I]) + ...);
        }

         /// Validates N-dimensional indices against bounds.
        template<std::uint64_t... I, typename... Indices>
        inline void checkBoundsUnrolled(std::index_sequence<I...>, Indices... idxs) const
        {
            if ((... || (static_cast<uint64_t>(idxs) >= _shape[I]))) 
            {
                throw std::out_of_range("Index out of range");
            }
        }

    public:
        /// Initializes tensor with a given shape.
        Tensor(std::vector<uint64_t> shape)
            : _shape(std::move(shape)),
              _strides(_shape.size(), 1)
        {
            if (_shape.empty())
            {
                throw std::invalid_argument("Shape must have at least one dimension");
            }
            
            // Compute strides (row-major order)
            for (uint64_t i = _shape.size(); i-- > 1; )
            {
                _strides[i - 1] = _strides[i] * _shape[i];
            }

            // Compute total size
            uint64_t totalSize = 1;
            for (uint64_t dim : _shape)
            {
                if (dim == 0)
                {
                    throw std::invalid_argument("Shape dimensions must be greater than 0");
                }
                totalSize *= dim;
            }

            _data.resize(totalSize);
        }

        /// Default destructor.
        ~Tensor() = default;

        /// Mutable element access with bounds checking.
        template<typename... Indices>
        T& operator()(Indices... idxs)
        {
            if (sizeof...(idxs) != _shape.size()) 
            {
                throw std::invalid_argument("Rank mismatch");
            }

            checkBoundsUnrolled(std::index_sequence_for<Indices...>{}, idxs...);
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)];
        }

        /// Const element access with bounds checking.
        template<typename... Indices>
        const T& operator()(Indices... idxs) const
        {
            if (sizeof...(idxs) != _shape.size()) 
            {
                throw std::invalid_argument("Rank mismatch");
            }

            checkBoundsUnrolled(std::index_sequence_for<Indices...>{}, idxs...);
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)];
        }

        /// Fast unchecked mutable element access.
        template<typename... Indices>
        inline T& unchecked(Indices... idxs)
        {
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)] ;          
        }

        /// Fast unchecked const element access.
        template<typename... Indices>
        inline const T& unchecked(Indices... idxs) const
        {
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)] ;          
        }

        /// Equality comparison.
        bool operator==(const Tensor<T>& otherTensor) const noexcept
        {
            return _shape == otherTensor._shape &&
                   _data == otherTensor._data;
        }

        /// Inequality comparison.
        bool operator!=(const Tensor<T>& otherTensor) const noexcept
        {
            return !(*this == otherTensor);
        }

        /// Element-wise addition.
        Tensor<T> operator+(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::plus<T>());
        }

        /// Element-wise subtraction.
        Tensor<T> operator-(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::minus<T>());
        }

        /// Element-wise multiplication.
        Tensor<T> operator*(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::multiplies<T>());
        }

        /// Element-wise division.
        Tensor<T> operator/(const Tensor<T>& otherTensor) const
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            if constexpr (std::is_floating_point<T>::value) 
            {
                return elementWiseOp(otherTensor, std::divides<T>());    
            }
            else            
            {
                for (const auto& val : otherTensor._data) 
                {
                    if (val == 0) 
                    {
                        throw std::invalid_argument("Can't divide by 0");
                    }                    
                }
                return elementWiseOp(otherTensor, std::divides<T>());
            }
        }

        /// Scalar multiplication.
        Tensor<T> operator*(const T& scalar) const
        {
            Tensor<T> result(_shape);
            std::transform(_data.begin(), 
                           _data.end(), 
                           result._data.begin(),
                           [&scalar](const T& val) { return val * scalar; });
            return result;
        }

        /// Scalar division.
        Tensor<T> operator/(const T& scalar) const
        {
            if (scalar == 0) 
            { 
                throw std::invalid_argument("Can't divide by 0"); 
            }
            
            Tensor<T> result(_shape);
            std::transform(_data.begin(), 
                           _data.end(), 
                           result._data.begin(),
                           [&scalar](const T& val) { return val / scalar; });
            return result;
        }

        /// In-place element-wise addition.
        Tensor<T>& operator+=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            std::transform(_data.begin(),
                           _data.end(),
                           otherTensor._data.begin(),
                           _data.begin(),
                           std::plus<T>());

            return *this;
        }
        
        /// In-place element-wise subtraction.
        Tensor<T>& operator-=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            std::transform(_data.begin(),
                           _data.end(),
                           otherTensor._data.begin(),
                           _data.begin(),
                           std::minus<T>());

            return *this;
        }

        /// In-place element-wise multiplication.
        Tensor<T>& operator*=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            std::transform(_data.begin(),
                           _data.end(),
                           otherTensor._data.begin(),
                           _data.begin(),
                           std::multiplies<T>());

            return *this;
        }

        /// In-place element-wise division.
        Tensor<T>& operator/=(const Tensor<T>& otherTensor)
        {
            if (_shape != otherTensor._shape) 
            { 
                throw std::invalid_argument("Tensor shape mismatch"); 
            }

            if constexpr (std::is_floating_point<T>::value) 
            {
                std::transform(_data.begin(), 
                               _data.end(), 
                               otherTensor._data.begin(), 
                               _data.begin(), 
                               std::divides<T>());
            }
            else            
            {
                for (const auto& val : otherTensor._data) 
                {
                    if (val == 0) 
                    {
                        throw std::invalid_argument("Can't divide by 0");
                    }                    
                }

                std::transform(_data.begin(), 
                               _data.end(), 
                               otherTensor._data.begin(), 
                               _data.begin(), 
                               std::divides<T>());
            }

            return *this;
        }

        /// In-place scalar multiplication.
        Tensor<T>& operator*=(const T& scalar)
        {
            std::transform(_data.begin(),
                           _data.end(),
                           _data.begin(),
                           [&scalar](const T& val){ return val * scalar; });
            return *this;
        }

        /// In-place scalar division.
        Tensor<T>& operator/=(const T& scalar)
        {
            if (scalar == 0) 
            { 
                throw std::invalid_argument("Can't divide by 0"); 
            }

            std::transform(_data.begin(),
                           _data.end(),
                           _data.begin(),
                           [&scalar](const T& val){ return val / scalar; });

            return *this;
        }

        /// Fills tensor with a specific value.
        template<typename U>
        void fill(const U& value)
        {
            static_assert(std::is_convertible<U, T>::value, "U must be convertible to T");
            std::fill(_data.begin(), _data.end(), static_cast<T>(value));
        }

        // --- Utilities ---

        /// Gets tensor dimension sizes.
        const std::vector<uint64_t>& shape() const noexcept { return _shape; }

        /// Gets tensor rank (number of dimensions).
        uint64_t rank() const noexcept { return _shape.size(); }
        
        /// Gets total element count.
        uint64_t size() const noexcept { return _data.size(); }

        /// Gets mutable pointer to raw data.
        T* data() noexcept { return _data.data(); }
        
        /// Gets const pointer to raw data.
        const T* data() const noexcept { return _data.data(); }
    };

    /// Left-hand scalar multiplication.
    template<typename T, typename U>
    Tensor<T> operator*(const U& scalar, const Tensor<T>& tensor)
    {
        return tensor * scalar;
    }

    /// Performs SIMD-optimized 2D matrix multiplication.
    template<typename T>
    void matmul(const Tensor<T>& srcTnsr1, const Tensor<T>& srcTnsr2, Tensor<T>& outTnsr)
    {
        if (srcTnsr1.shape().size() != 2 || srcTnsr2.shape().size() != 2) 
        { 
            throw std::runtime_error("matmul requires matrices (2D tensors)."); 
        }
        if (srcTnsr1.shape()[1] !=srcTnsr2.shape()[0])
        {
            throw std::invalid_argument("matmul dimension mismatch");
        }
        if (outTnsr.shape()[0] != srcTnsr1.shape()[0] || outTnsr.shape()[1] != srcTnsr2.shape()[1]) 
        {
            throw std::runtime_error("result tensor shape mismatch");
        }
        uint64_t r1 = srcTnsr1.shape()[0];
        uint64_t c1 = srcTnsr1.shape()[1];
        uint64_t c2 = srcTnsr2.shape()[1];
        const T* st1Ptr = srcTnsr1.data();
        const T* st2Ptr = srcTnsr2.data();
        T* otPtr = outTnsr.data();
        outTnsr.fill(T{0});

        if constexpr (std::is_same_v<T, float>) 
        {
            uint64_t vecEnd = (c2 / 8) * 8;
            uint64_t iLim = (r1 / 4) * 4;

            for (uint64_t i = 0; i < iLim; i += 4) 
            {
                const T* st1Row0 = st1Ptr + ((i + 0) * c1);
                const T* st1Row1 = st1Ptr + ((i + 1) * c1);
                const T* st1Row2 = st1Ptr + ((i + 2) * c1);
                const T* st1Row3 = st1Ptr + ((i + 3) * c1);

                T* otRow0 = otPtr + ((i + 0) * c2);
                T* otRow1 = otPtr + ((i + 1) * c2);
                T* otRow2 = otPtr + ((i + 2) * c2);
                T* otRow3 = otPtr + ((i + 3) * c2);

                for (uint64_t j = 0; j < vecEnd; j += 8) 
                {
                    __m256 otVec0 = _mm256_loadu_ps(&otRow0[j]);
                    __m256 otVec1 = _mm256_loadu_ps(&otRow1[j]);
                    __m256 otVec2 = _mm256_loadu_ps(&otRow2[j]);
                    __m256 otVec3 = _mm256_loadu_ps(&otRow3[j]);
                    
                    for (uint64_t k = 0; k < c1; ++k) 
                    {
                        __m256 st2Vec = _mm256_loadu_ps(&st2Ptr[k * c2  + j]);

                        __m256 st1Sclr0 = _mm256_set1_ps(st1Row0[k]);
                        __m256 st1Sclr1 = _mm256_set1_ps(st1Row1[k]);
                        __m256 st1Sclr2 = _mm256_set1_ps(st1Row2[k]);
                        __m256 st1Sclr3 = _mm256_set1_ps(st1Row3[k]);

                        otVec0 = _mm256_fmadd_ps(st1Sclr0, st2Vec, otVec0); 
                        otVec1 = _mm256_fmadd_ps(st1Sclr1, st2Vec, otVec1); 
                        otVec2 = _mm256_fmadd_ps(st1Sclr2, st2Vec, otVec2); 
                        otVec3 = _mm256_fmadd_ps(st1Sclr3, st2Vec, otVec3); 
                    }
                    
                    _mm256_storeu_ps(&otRow0[j], otVec0);
                    _mm256_storeu_ps(&otRow1[j], otVec1);
                    _mm256_storeu_ps(&otRow2[j], otVec2);
                    _mm256_storeu_ps(&otRow3[j], otVec3);
                }
                for (uint64_t j = vecEnd; j < c2; ++j) 
                {   
                    for (uint64_t k = 0; k < c1; ++k)
                    {                        
                        otRow0[j] += st1Row0[k] * st2Ptr[k * c2 + j];
                        otRow1[j] += st1Row1[k] * st2Ptr[k * c2 + j];
                        otRow2[j] += st1Row2[k] * st2Ptr[k * c2 + j];
                        otRow3[j] += st1Row3[k] * st2Ptr[k * c2 + j];

                    }
                }
            }
            for (uint64_t i = iLim; i < r1; i += 1) 
            {
                const T* st1Row = st1Ptr + (i * c1);
                T* otRow = otPtr + (i * c2);

                for (uint64_t k = 0; k < c1; ++k) 
                {
                    T st1Sclr = st1Row[k]; 
                    const T* st2Row = st2Ptr + (k * c2);
                    
                    __m256 aSclr = _mm256_set1_ps(st1Sclr);
                    
                    for (uint64_t j = 0; j < vecEnd; j += 8) 
                    {
                        __m256 st2Vec = _mm256_loadu_ps(&st2Row[j]);
                        __m256 otVec = _mm256_loadu_ps(&otRow[j]);
                        otVec = _mm256_fmadd_ps(aSclr, st2Vec, otVec);
                        _mm256_storeu_ps(&otRow[j], otVec);
                    }
                    
                    for (uint64_t j = vecEnd; j < c2; ++j) 
                    {
                        otRow[j] += st1Sclr * st2Row[j];
                    }
                }
            }
        }
        else if constexpr (std::is_same_v<T, double>) 
        {
            uint64_t vecEnd = (c2 / 4) * 4;
            uint64_t iLim = (r1 / 4) * 4;

            for (uint64_t i = 0; i < iLim; i += 4) 
            {
                const T* aRow0 = st1Ptr + ((i + 0) * c1);
                const T* aRow1 = st1Ptr + ((i + 1) * c1);
                const T* aRow2 = st1Ptr + ((i + 2) * c1);
                const T* aRow3 = st1Ptr + ((i + 3) * c1);

                T* cRow0 = otPtr + ((i + 0) * c2);
                T* cRow1 = otPtr + ((i + 1) * c2);
                T* cRow2 = otPtr + ((i + 2) * c2);
                T* cRow3 = otPtr + ((i + 3) * c2);

                for (uint64_t j = 0; j < vecEnd; j += 4) 
                {
                    __m256d cVec0 = _mm256_loadu_pd(&cRow0[j]);
                    __m256d cVec1 = _mm256_loadu_pd(&cRow1[j]);
                    __m256d cVec2 = _mm256_loadu_pd(&cRow2[j]);
                    __m256d cVec3 = _mm256_loadu_pd(&cRow3[j]);
                    
                    for (uint64_t k = 0; k < c1; ++k) 
                    {
                        __m256d bVec = _mm256_loadu_pd(&st2Ptr[k * c2  + j]);

                        __m256d aSclr0 = _mm256_set1_pd(aRow0[k]);
                        __m256d aSclr1 = _mm256_set1_pd(aRow1[k]);
                        __m256d aSclr2 = _mm256_set1_pd(aRow2[k]);
                        __m256d aSclr3 = _mm256_set1_pd(aRow3[k]);

                        cVec0 = _mm256_fmadd_pd(aSclr0, bVec, cVec0); 
                        cVec1 = _mm256_fmadd_pd(aSclr1, bVec, cVec1); 
                        cVec2 = _mm256_fmadd_pd(aSclr2, bVec, cVec2); 
                        cVec3 = _mm256_fmadd_pd(aSclr3, bVec, cVec3); 
                    }
                    
                    _mm256_storeu_pd(&cRow0[j], cVec0);
                    _mm256_storeu_pd(&cRow1[j], cVec1);
                    _mm256_storeu_pd(&cRow2[j], cVec2);
                    _mm256_storeu_pd(&cRow3[j], cVec3);
                }
                for (uint64_t j = vecEnd; j < c2; ++j) 
                {   
                    for (uint64_t k = 0; k < c1; ++k)
                    {                        
                        cRow0[j] += aRow0[k] * st2Ptr[k * c2 + j];
                        cRow1[j] += aRow1[k] * st2Ptr[k * c2 + j];
                        cRow2[j] += aRow2[k] * st2Ptr[k * c2 + j];
                        cRow3[j] += aRow3[k] * st2Ptr[k * c2 + j];

                    }
                }
            }
            for (uint64_t i = iLim; i < r1; i += 1) 
            {
                 const T* st1Row = st1Ptr + (i * c1);
                T* otRow = otPtr + (i * c2);

                for (uint64_t k = 0; k < c1; ++k) 
                {
                    T st1Sclr = st1Row[k]; 
                    const T* st2Row = st2Ptr + (k * c2);
                    
                    __m256d aSclr = _mm256_set1_pd(st1Sclr);
                    
                    for (uint64_t j = 0; j < vecEnd; j += 4) 
                    {
                        __m256d st2Vec = _mm256_loadu_pd(&st2Row[j]);
                        __m256d otVec = _mm256_loadu_pd(&otRow[j]);
                        otVec = _mm256_fmadd_pd(aSclr, st2Vec, otVec);
                        _mm256_storeu_pd(&otRow[j], otVec);
                    }
                    
                    for (uint64_t j = vecEnd; j < c2; ++j) 
                    {
                        otRow[j] += st1Sclr * st2Row[j];
                    }
                }
        }
        }
        else
        {
            for (uint64_t i = 0; i < r1; ++i) 
            {
                const T* st1Row = st1Ptr + (i * c1);
                T* otRow = otPtr + (i * c2);
                for (uint64_t k = 0; k < c1; ++k) 
                {
                    T st1Sclr = st1Row[k]; 
                    const T* st2Row = st2Ptr + (k * c2);
                    for (uint64_t j = 0; j < c2; ++j) 
                    {
                        otRow[j] += st1Sclr * st2Row[j];
                    }
                }
            }
        }
    }

    /// Performs SIMD-optimized matrix-vector multiplication.
    template<typename T>
    void matvec(const Tensor<T>& srcTnsr, const Tensor<T>& srcVec, Tensor<T>& outVec)
    {
        if (srcTnsr.shape().size() != 2 || srcVec.shape().size() != 1) 
        {
            throw std::invalid_argument("matvec requires matrix & tensor");
        }
        if (srcTnsr.shape()[1] != srcVec.shape()[0]) 
        {
            throw std::invalid_argument("matvec dimension mismatch");
        }
        if (outVec.shape()[0] != srcTnsr.shape()[0]) 
        {
            throw std::invalid_argument("result vector size mismatch");
        }

        uint64_t r1 = srcTnsr.shape()[0];
        uint64_t r2 = srcVec.shape()[0];
        const T* stPtr = srcTnsr.data();
        const T* svPtr = srcVec.data();
        T* outPtr = outVec.data();
        
        outVec.fill(T{0});

        if constexpr (std::is_same_v<T, float>) 
        {
            uint64_t vecEnd = (r2 / 8) * 8;
            uint64_t iLim = (r1 / 4) * 4;

            for (uint64_t i = 0; i < iLim; i += 4) 
            {
                T tmp[8];
                T sum0{0}, sum1{0}, sum2{0}, sum3{0};
                
                const T* stRow0 = stPtr + ((i + 0) * r2);
                const T* stRow1 = stPtr + ((i + 1) * r2);
                const T* stRow2 = stPtr + ((i + 2) * r2);
                const T* stRow3 = stPtr + ((i + 3) * r2);

                __m256 acc0 = _mm256_setzero_ps(); 
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (uint64_t j = 0; j < vecEnd; j += 8) 
                {
                    __m256 vecVec = _mm256_loadu_ps(&svPtr[j]);

                    __m256 stRowVec0 = _mm256_loadu_ps(&stRow0[j]);
                    __m256 stRowVec1 = _mm256_loadu_ps(&stRow1[j]);
                    __m256 stRowVec2 = _mm256_loadu_ps(&stRow2[j]);
                    __m256 stRowVec3 = _mm256_loadu_ps(&stRow3[j]);
                    
                    acc0 = _mm256_fmadd_ps(stRowVec0, vecVec, acc0);
                    acc1 = _mm256_fmadd_ps(stRowVec1, vecVec, acc1);
                    acc2 = _mm256_fmadd_ps(stRowVec2, vecVec, acc2);
                    acc3 = _mm256_fmadd_ps(stRowVec3, vecVec, acc3);
                }
                _mm256_storeu_ps(tmp, acc0);
                sum0 = tmp[0] + tmp[1] + tmp[2] + tmp[3] + 
                       tmp[4] + tmp[5] + tmp[6] + tmp[7];

                _mm256_storeu_ps(tmp, acc1);
                sum1 = tmp[0] + tmp[1] + tmp[2] + tmp[3] + 
                       tmp[4] + tmp[5] + tmp[6] + tmp[7];

                _mm256_storeu_ps(tmp, acc2);
                sum2 = tmp[0] + tmp[1] + tmp[2] + tmp[3] + 
                       tmp[4] + tmp[5] + tmp[6] + tmp[7];

                _mm256_storeu_ps(tmp, acc3);
                sum3 = tmp[0] + tmp[1] + tmp[2] + tmp[3] + 
                       tmp[4] + tmp[5] + tmp[6] + tmp[7];

                for (uint64_t j = vecEnd; j < r2; ++j) 
                {
                    sum0 += stRow0[j] * svPtr[j];
                    sum1 += stRow1[j] * svPtr[j];
                    sum2 += stRow2[j] * svPtr[j];
                    sum3 += stRow3[j] * svPtr[j];
                }

                outPtr[i + 0] = sum0;
                outPtr[i + 1] = sum1;
                outPtr[i + 2] = sum2;
                outPtr[i + 3] = sum3;

            }
            
            for (uint64_t i = iLim; i < r1; ++i) 
            {
                const T* aRow = stPtr + (i * r2);
                T sum = 0;
                for (uint64_t j = 0; j < r2; ++j) 
                {
                    sum += aRow[j] * svPtr[j];
                }
                outPtr[i] = sum;
            }
        }
        else if constexpr (std::is_same_v<T, double>) 
        {
            uint64_t vecEnd = (r2 /4) * 4;
            uint64_t iLim = (r1 / 4) * 4;

            for (uint64_t i = 0; i < iLim; i += 4) 
            {
                T tmp[4];
                T sum0{0}, sum1{0}, sum2{0}, sum3{0};

                const T* stRow0 = stPtr + ((i + 0) * r2);
                const T* stRow1 = stPtr + ((i + 1) * r2);
                const T* stRow2 = stPtr + ((i + 2) * r2);
                const T* stRow3 = stPtr + ((i + 3) * r2);

                __m256d acc0 = _mm256_setzero_pd(); 
                __m256d acc1 = _mm256_setzero_pd();
                __m256d acc2 = _mm256_setzero_pd();
                __m256d acc3 = _mm256_setzero_pd();

                for (uint64_t j = 0; j < vecEnd; j += 4) 
                {
                    __m256d vecVec = _mm256_loadu_pd(&svPtr[j]);

                    __m256d stRowVec0 = _mm256_loadu_pd(&stRow0[j]);
                    __m256d stRowVec1 = _mm256_loadu_pd(&stRow1[j]);
                    __m256d stRowVec2 = _mm256_loadu_pd(&stRow2[j]);
                    __m256d stRowVec3 = _mm256_loadu_pd(&stRow3[j]);
                    
                    acc0 = _mm256_fmadd_pd(stRowVec0, vecVec, acc0);
                    acc1 = _mm256_fmadd_pd(stRowVec1, vecVec, acc1);
                    acc2 = _mm256_fmadd_pd(stRowVec2, vecVec, acc2);
                    acc3 = _mm256_fmadd_pd(stRowVec3, vecVec, acc3);
                }

                _mm256_storeu_pd(tmp, acc0);
                sum0 = tmp[0] + tmp[1] + tmp[2] +tmp[3];

                _mm256_storeu_pd(tmp, acc1);
                sum1 = tmp[0] + tmp[1] + tmp[2] +tmp[3];

                _mm256_storeu_pd(tmp, acc2);
                sum2 = tmp[0] + tmp[1] + tmp[2] +tmp[3];
                 
                _mm256_storeu_pd(tmp, acc3);
                sum3 = tmp[0] + tmp[1] + tmp[2] +tmp[3]; 

                for (uint64_t j = vecEnd; j < r2; ++j) 
                {
                    sum0 += stRow0[j] * svPtr[j];
                    sum1 += stRow1[j] * svPtr[j];
                    sum2 += stRow2[j] * svPtr[j];
                    sum3 += stRow3[j] * svPtr[j];
                }

                outPtr[i + 0] = sum0;
                outPtr[i + 1] = sum1;
                outPtr[i + 2] = sum2;
                outPtr[i + 3] = sum3;

            }
            
            for (uint64_t i = iLim; i < r1; ++i) 
            {
                const T* aRow = stPtr + (i * r2);
                T sum = 0;
                for (uint64_t j = 0; j < r2; ++j) 
                {
                    sum += aRow[j] * svPtr[j];
                }
                outPtr[i] = sum;
            }
        }
        else 
        {
            for (uint64_t i = 0; i < r1; ++i) 
            {
                const T* aRow = stPtr + (i * r2);
                T sum = 0;
                for (uint64_t j = 0; j < r2; ++j) 
                {
                    sum += aRow[j] * svPtr[j];
                }
                outPtr[i] = sum;
            }
        }
    }

    /// Transposes a 2D matrix (supports in-place for square matrices).
    template<typename T>
    void transpose(Tensor<T>& a, Tensor<T>& b)
    {
        if (a.shape().size() != 2 || b.shape().size() != 2) 
        { 
            throw std::runtime_error("Transposition only supports matrices for now"); 
        }

        if ((a.shape()[0] == a.shape()[1]) && (&a == &b)) 
        {
            uint64_t n = a.shape()[0];
            for(uint64_t i = 0; i < n; ++i)
            {
                for(uint64_t j = i + 1; j < n; ++j)
                {
                    std::swap(a.unchecked(i, j), a.unchecked(j, i));
                }
            }
        }
        else 
        {
            uint64_t rows = a.shape()[0], cols = a.shape()[1];

            for (uint64_t i = 0; i < rows; ++i)
            {
                for (uint64_t j = 0; j < cols; ++j)
                {
                    b.unchecked(j, i) = a.unchecked(i, j);
                }
            }
        }
    }
}