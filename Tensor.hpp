/**
 * @file Tensor.hpp
 * @brief Lightweight generic N-dimensional tensor (matrix) implementation.
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
    /**
     * @class Tensor
     * @brief A generic N-dimensional tensor supporting element-wise operations.
     *
     * Provides:
     * - Basic element access
     * - Element-wise arithmetic
     * - Scalar operations
     * - Matrix transpose (for rank-2 tensors)
     *
     * @tparam T Element type (must be arithmetic).
     */
    template<typename T>
    class Tensor
    {
        static_assert(std::is_arithmetic<T>::value, "Type must be numeric");

    private:
        std::vector<uint64_t> _shape;       ///< Dimensions of the tensor.
        std::vector<uint64_t> _strides;     ///< Stride values for flat indexing.
        std::vector<T> _data;             ///< Contiguous storage for tensor data.

        /**
         * @brief Apply an element-wise binary operation between two tensors.
         * @tparam BinaryOp Binary operation type (e.g., std::plus, std::multiplies).
         * @param otherTensor The other tensor to combine with.
         * @param op Binary operation functor.
         * @return New tensor containing the result.
         * @throws std::invalid_argument if shapes do not match.
         */
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

        /**
         * @brief Compute the flattened (linear) index from N-dimensional indices.
         * @tparam I Compile-time index sequence corresponding to tensor dimensions.
         * @tparam Indices Variadic index argument types.
         * @param idxs N-dimensional indices.
         * @return Flattened index into the underlying contiguous storage.
         */
        template<std::uint64_t... I, typename... Indices>
        inline uint64_t computeFlatUnrolled(std::index_sequence<I...>, Indices... idxs) const
        {
            return ((static_cast<uint64_t>(idxs) * _strides[I]) + ...);
        }

         /**
         * @brief Validate N-dimensional indices against tensor bounds.
         * @tparam I Compile-time index sequence corresponding to tensor dimensions.
         * @tparam Indices Variadic index argument types.
         * @param idxs N-dimensional indices.
         * @throws std::out_of_range if any index exceeds its dimension bounds.
         */
        template<std::uint64_t... I, typename... Indices>
        inline void checkBoundsUnrolled(std::index_sequence<I...>, Indices... idxs) const
        {
            if ((... || (static_cast<uint64_t>(idxs) >= _shape[I]))) 
            {
                throw std::out_of_range("Index out of range");
            }
        }

    public:
        /**
         * @brief Construct a tensor of given shape with default-initialized elements.
         * @param shape Vector specifying the size of each dimension.
         * @throws std::invalid_argument if any dimension is zero.
         */
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

        /**
         * @brief Mutable element access.
         * @tparam Indices Variadic list of index arguments.
         * @param idxs N-dimensional indices.
         * @return Reference to the element.
         * @throws std::invalid_argument if number of indicies given doesn't match tensor's. 
         * @throws std::out_of_range if any index is invalid.
         */
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

        /**
         * @brief Const element access.
         * @tparam Indices Variadic list of index arguments.
         * @param idxs N-dimensional indices.
         * @return Const reference to the element.
         * @throws std::invalid_argument if number of indicies given doesn't match tensor's. 
         * @throws std::out_of_range if any index is invalid.
         */
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

        /**
        * @brief Fast unchecked element access.
        * @tparam Indices Variadic list of index arguments.
        * @param idxs N-dimensional indices.
        * @return Reference to the selected element.
        *
        * @note Out-of-range indices result in undefined behavior.
        */
        template<typename... Indices>
        inline T& unchecked(Indices... idxs)
        {
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)] ;          
        }

        template<typename... Indices>
        inline const T& unchecked(Indices... idxs) const
        {
            return _data[computeFlatUnrolled(std::index_sequence_for<Indices...>{}, idxs...)] ;          
        }

        /**
        * @brief Compare two tensors for equality.
        * @param otherTensor The tensor to compare with.
        * @return true if shapes and data match exactly, false otherwise.
        */
        bool operator==(const Tensor<T>& otherTensor) const noexcept
        {
            return _shape == otherTensor._shape &&
                   _data == otherTensor._data;
        }

        /**
        * @brief Compare two tensors for inequality.
        * @param otherTensor The tensor to compare with.
        * @return true if tensors differ, false otherwise.
        */
        bool operator!=(const Tensor<T>& otherTensor) const noexcept
        {
            return !(*this == otherTensor);
        }

        /**
        * @brief Element-wise addition.
        * @param otherTensor Tensor to add.
        * @return Resulting tensor after element-wise addition.
        * @throws std::invalid_argument if shapes do not match.
        */
        Tensor<T> operator+(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::plus<T>());
        }

        /**
        * @brief Element-wise subtraction.
        * @param otherTensor Tensor to subtract.
        * @return New tensor containing element-wise differences.
        * @throws std::invalid_argument if shapes do not match.
        */
        Tensor<T> operator-(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::minus<T>());
        }

        /**
        * @brief Element-wise multiplication.
        * @param otherTensor Tensor to multiply with.
        * @return Tensor containing element-wise products.
        * @throws std::invalid_argument if shapes do not match.
        */
        Tensor<T> operator*(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::multiplies<T>());
        }

        /**
        * @brief Element-wise division.
        * @param otherTensor Tensor providing divisors.
        * @return Tensor containing element-wise quotients.
        * @throws std::invalid_argument if shapes differ or any divisor is zero.
        */
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

        /**
        * @brief Multiply all elements by a scalar.
        * @param scalar Scalar multiplier.
        * @return New tensor after scalar multiplication.
        */
        Tensor<T> operator*(const T& scalar) const
        {
            Tensor<T> result(_shape);
            std::transform(_data.begin(), 
                           _data.end(), 
                           result._data.begin(),
                           [&scalar](const T& val) { return val * scalar; });
            return result;
        }

        /**
        * @brief Divide all elements by a scalar.
        * @param scalar Divisor.
        * @return New tensor after scalar division.
        * @throws std::invalid_argument if scalar is zero.
        */
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

        /**
        * @brief In-place element-wise addition.
        * @param otherTensor Tensor to add.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if shapes do not match.
        */
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
        
        /**
        * @brief In-place element-wise subtraction.
        * @param otherTensor Tensor to subtract.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if shapes do not match.
        */
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

        /**
        * @brief In-place element-wise multiplication.
        * @param otherTensor Tensor to multiply with.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if shapes do not match.
        */
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

        /**
        * @brief In-place element-wise division.
        * @param otherTensor Tensor providing divisors.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if shapes differ or any divisor is zero.
        */
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

        /**
        * @brief In-place scalar multiplication.
        * @param scalar Multiplier.
        * @return Reference to this tensor.
        */
        Tensor<T>& operator*=(const T& scalar)
        {
            std::transform(_data.begin(),
                           _data.end(),
                           _data.begin(),
                           [&scalar](const T& val){ return val * scalar; });
            return *this;
        }

        /**
        * @brief In-place scalar division.
        * @param scalar Divisor.
        * @return Reference to this tensor.
        * @throws std::invalid_argument if scalar is zero.
        */
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

        /**
         * @brief Fill the tensor with a specified value.
         * @tparam U Type convertible to T.
         * @param value Value to fill.
         */
        template<typename U>
        void fill(const U& value)
        {
            static_assert(std::is_convertible<U, T>::value, "U must be convertible to T");
            std::fill(_data.begin(), _data.end(), static_cast<T>(value));
        }

        // --- Utilities ---

        /**
        * @brief Get the tensor's shape.
        * @return Vector of dimension sizes.
        */
        const std::vector<uint64_t>& shape() const noexcept { return _shape; }

        /**
        * @brief Get the number of dimensions.
        * @return Tensor rank.
        */
        uint64_t rank() const noexcept { return _shape.size(); }
        
        /**
        * @brief Get total number of stored elements.
        * @return Element count.
        */
        uint64_t size() const noexcept { return _data.size(); }

        /**
         * @brief Get a pointer to the underlying contiguous data storage.
         * @return Pointer to the first element in the tensor.
         */
        T* data() noexcept { return _data.data(); }
        
        /**
         * @brief Get a const pointer to the underlying contiguous data storage.
         * @return Const pointer to the first element in the tensor.
         */
        const T* data() const noexcept { return _data.data(); }
    };

    /**
     * @brief Scalar-tensor multiplication (scalar on left-hand side).
     * @tparam T Tensor element type.
     * @tparam U Scalar type.
     * @param scalar Scalar value.
     * @param tensor Tensor object.
     * @return New tensor after scalar multiplication.
     */
    template<typename T, typename U>
    Tensor<T> operator*(const U& scalar, const Tensor<T>& tensor)
    {
        return tensor * scalar;
    }

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
        const T* aPtr = srcTnsr1.data();
        const T* bPtr = srcTnsr2.data();
        T* cPtr = outTnsr.data();
        
        outTnsr.fill(T{0});
        
        for (uint64_t i = 0; i < r1; ++i) 
        {
            const T* aRow = aPtr + (i * c1);
            T* cRow = cPtr + (i * c2);
            for (uint64_t k = 0; k < c1; ++k) 
            {
                T aIk = aRow[k]; 
                const T* bRow = bPtr + (k * c2);
                for (uint64_t j = 0; j < c2; ++j) 
                {
                    cRow[j] += aIk * bRow[j];
                }
            }
        }
    }

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
        const T* aPtr = srcTnsr.data();
        const T* bPtr = srcVec.data();
        T* cPtr = outVec.data();
        
        outVec.fill(T{0});

        for (uint64_t i = 0; i < r1; ++i) 
        {
            const T* aRow = aPtr + (i * r2);
            T sum = 0;
            for (uint64_t j = 0; j < r2; ++j) 
            {
                sum += aRow[j] * bPtr[j];
            }
            cPtr[i] = sum;
        }
    }

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