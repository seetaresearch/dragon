// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_CUDNN_DEVICE_H_
#define DRAGON_UTILS_CUDNN_DEVICE_H_

#ifdef WITH_CUDNN

#include <stdint.h>
#include <vector>
#include <cudnn.h>

#include "core/typeid.h"

namespace dragon {

class Tensor;

#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_VERSION_MAX(major, minor, patch) \
    (CUDNN_VERSION < (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << "\n" \
      << cudnnGetErrorString(status); \
    } while (0)

template <typename T> class CUDNNType;

template<> class CUDNNType<float>  {
 public:
    static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
    static float oneval, zeroval;
    static const void *one, *zero;
};

template<> class CUDNNType<double> {
 public:
    static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
    static double oneval, zeroval;
    static const void *one, *zero;
};

#ifdef WITH_CUDA_FP16
template<> class CUDNNType<float16> {
 public:
    static const cudnnDataType_t type = CUDNN_DATA_HALF;
    static float oneval, zeroval;
    static const void *one, *zero;
};
#endif

template <typename T>
void cudnnSetTensorDesc(cudnnTensorDescriptor_t* desc, Tensor* tensor);

template <typename T>
void cudnnSetTensor4dDesc(cudnnTensorDescriptor_t* desc, const string& data_format, Tensor* tensor);

template <typename T>
void cudnnSetTensor5dDesc(cudnnTensorDescriptor_t* desc, const string& data_format, Tensor* tensor);

template <typename T>
void cudnnSetTensor3dDesc(cudnnTensorDescriptor_t* desc, const string& data_format, Tensor* tensor);

template <typename T>
void cudnnSetTensorDesc(cudnnTensorDescriptor_t* desc, const std::vector<int64_t>& dims);

template <typename T>
void cudnnSetTensor4dDesc(cudnnTensorDescriptor_t* desc, const string& data_format, const std::vector<int64_t>& dims);

template <typename T>
void cudnnSetTensor5dDesc(cudnnTensorDescriptor_t* desc, const string& data_format, const std::vector<int64_t>& dims);

template <typename T>
void cudnnSetTensor3dDesc(cudnnTensorDescriptor_t* desc, const string& data_format, const std::vector<int64_t>& dims);

template <typename T>
void cudnnSetTensorDesc(cudnnTensorDescriptor_t* desc,
                        const std::vector<int64_t>& dims,
                        const std::vector<int64_t>& strides);

}

#endif    // WITH_CUDNN

#endif    // DRAGON_UTILS_CUDNN_DEVICE_H_