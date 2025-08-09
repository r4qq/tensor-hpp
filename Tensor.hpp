/**
 * @file Tensor.hpp
 * @brief Lightweight generic 2D tensor (matrix) implementation.
 * @author r4qq
 * @date 2025
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Tensor
{
    
    template<typename T>
    class Tensor
    {
        static_assert(std::is_arithmetic<T>::value, "Type must be numeric");

    private:
        std::vector<size_t> _shape;       
        std::vector<size_t> _strides;
        std::vector<T> _data;           

        template<typename BinaryOp>
        inline Tensor<T> elementWiseOp(const Tensor<T>& otherTensor, BinaryOp op) const
        {
            if (_shape != otherTensor._shape) 
                throw std::invalid_argument("Tensor shape mismatch");

            Tensor<T> result(_shape);

            std::transform(_data.begin(), 
                           _data.end(), 
                           otherTensor._data.begin(), 
                           result._data.begin(),
                           op);      

            return result;     
        }

        template<size_t N>
        size_t computeFlatIndex(const std::array<size_t, N>& indices) const
        {
            size_t flatIndex = 0;
            for(size_t i = 0; i < indices.size(); i++)
            {
                flatIndex += _strides[i] * indices[i];
            }

            return flatIndex;
        }

    public:
       
        Tensor(std::vector<size_t> shape):
            _shape(std::move(shape)), 
            _strides(_shape.size(), 1),
            _data([&]()
            {    
                // Strides (row-major)
                for (size_t i = shape.size(); i-- > 1; )
                {
                    _strides[i - 1] = _strides[i] * _shape[i];
                }

                // Compute total size
                size_t totalSize = 1;
                for (size_t dim : _shape)
                {
                    if (dim == 0)
                        throw std::invalid_argument("Shape dimensions must be greater than 0");
                    totalSize *= dim;
                }
                return totalSize;
            })  
            {
                //empty constructor body - all done in initializer list
            }

        ~Tensor() = default;

        template<typename... Indices>
        T& operator()(Indices... indices)
        {
            static_assert(sizeof...(indices) == _shape.size(), "Invalid number of indices");
            std::array<size_t, sizeof...(indices)> idxArr{static_cast<size_t>(indices)...};
            for(size_t i = 0; i < _shape.size(); i++)
            {
                if(idxArr[i] >= _shape[i])
                    throw std::out_of_range("Index" + std::to_string(idxArr[i]) + " is out of range");
            }

            return _data[computeFlatIndex(idxArr)];
        }

        template<typename... Indices>
        const T& operator()(Indices... indices) const
        {
            static_assert(sizeof...(indices) == _shape.size(), "Invalid number of indices");
            std::array<size_t, sizeof...(indices)> idxArr{static_cast<size_t>(indices)...};
            for(size_t i = 0; i < _shape.size(); i++)
            {
                if(idxArr[i] >= _shape[i])
                    throw std::out_of_range("Index" + std::to_string(idxArr[i]) + " is out of range");
            }

            return _data[computeFlatIndex(idxArr)];
        }
       
        bool operator==(const Tensor<T>& otherTensor) const
        {
            return _shape == otherTensor._shape && 
                   _strides == otherTensor._strides && 
                   _data == otherTensor._data;
        }
 
        bool operator!=(const Tensor<T>& otherTensor) const
        {
            return !(*this == otherTensor);
        }

        Tensor<T> operator+(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::plus<T>());
        }

        Tensor<T> operator-(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::minus<T>());
        }

     
        Tensor<T> operator*(const T& scalar) const
        {
            Tensor<T> result(_shape);
            std::transform(_data.begin(), _data.end(), result._data.begin(),
                           [&scalar](const T& val){ return val * scalar; });
            return result;
        }

        //element-wise multiplication
        Tensor<T> operator*(const Tensor<T>& otherTensor) const
        {
            return elementWiseOp(otherTensor, std::multiplies<T>());
        }

        Tensor<T> matmul(const Tensor<T>& otherTensor) const
        {
            //todo matrix multiplication
        }
      
        Tensor<T> transpose() const
        {
            static_assert(_shape.size() == 2, "Trasnposition only supports matrices for now");

            size_t rows = _shape[0], cols = _shape[1];
            Tensor<T> result({cols, rows});

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(j, i) = (*this)(i, j);

            return result;
        }

        template<typename U>
        void fill(const U& value)
        {
            static_assert(std::is_convertible<U, T>::value, "U must be convertible to T");
            std::fill(_data.begin(), _data.end(), static_cast<T>(value));
        }

    };
   
    template<typename T, typename U>
    Tensor<T> operator*(const U& scalar, const Tensor<T>& tensor)
    {
        return tensor * scalar;
    }
}
