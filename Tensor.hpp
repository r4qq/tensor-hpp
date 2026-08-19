/**
 * @file Tensor.hpp
 * @brief Lightweight generic N-dimensional tensor.
 * @author r4qq 
 * @date 2025-2026
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

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

        Tensor(const Tensor&) = default;
        Tensor(Tensor&&) = default;
        Tensor& operator=(const Tensor&) = default;
        Tensor& operator=(Tensor&&) = default;
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

    /// Performs 2D matrix multiplication.
    template<typename T>
    void matmul(const Tensor<T>& srcTnsr1, const Tensor<T>& srcTnsr2, Tensor<T>& outTnsr)
    {
        if (srcTnsr1.shape().size() != 2 || srcTnsr2.shape().size() != 2) 
        { 
            throw std::invalid_argument("matmul requires matrices (2D tensors)."); 
        }
        if (srcTnsr1.shape()[1] !=srcTnsr2.shape()[0])
        {
            throw std::invalid_argument("matmul dimension mismatch");
        }
        if (outTnsr.shape()[0] != srcTnsr1.shape()[0] || outTnsr.shape()[1] != srcTnsr2.shape()[1]) 
        {
            throw std::invalid_argument("result tensor shape mismatch");
        }
        uint64_t r1 = srcTnsr1.shape()[0];
        uint64_t c1 = srcTnsr1.shape()[1];
        uint64_t c2 = srcTnsr2.shape()[1];
        const T* st1Ptr = srcTnsr1.data();
        const T* st2Ptr = srcTnsr2.data();
        T* otPtr = outTnsr.data();
                
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

    /// Performs matrix-vector multiplication.
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
        T* ovPtr = outVec.data();
        
        for (uint64_t i = 0; i < r1; ++i) 
        {
            const T* stRow = stPtr + (i * r2);
            T sum = 0;
            for (uint64_t j = 0; j < r2; ++j) 
            {
                sum += stRow[j] * svPtr[j];
            }
            ovPtr[i] = sum;
        }
    }

    /// Transposes a 2D matrix (supports in-place for square matrices).
    template<typename T>
    void transpose(Tensor<T>& srcTnsr, Tensor<T>& outTnsr)
    {
        if (srcTnsr.shape().size() != 2 || outTnsr.shape().size() != 2) 
        { 
            throw std::runtime_error("Transposition only supports matrices for now"); 
        }

        if ((srcTnsr.shape()[0] == srcTnsr.shape()[1]) && (&srcTnsr == &outTnsr)) 
        {
            uint64_t n = srcTnsr.shape()[0];
            for(uint64_t i = 0; i < n; ++i)
            {
                for(uint64_t j = i + 1; j < n; ++j)
                {
                    std::swap(srcTnsr.unchecked(i, j), srcTnsr.unchecked(j, i));
                }
            }
        }
        else 
        {
            uint64_t rows = srcTnsr.shape()[0], cols = srcTnsr.shape()[1];

            for (uint64_t i = 0; i < rows; ++i)
            {
                for (uint64_t j = 0; j < cols; ++j)
                {
                    outTnsr.unchecked(j, i) = srcTnsr.unchecked(i, j);
                }
            }
        }
    }
}