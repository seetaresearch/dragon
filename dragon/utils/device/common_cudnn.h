/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_DEVICE_COMMON_CUDNN_H_
#define DRAGON_UTILS_DEVICE_COMMON_CUDNN_H_

#ifdef USE_CUDNN

#include <cudnn.h>

#include "dragon/core/common.h"

namespace dragon {

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_VERSION_MAX(major, minor, patch) \
  (CUDNN_VERSION < (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition)                                             \
  do {                                                                     \
    cudnnStatus_t status = condition;                                      \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << "\n"                         \
                                           << cudnnGetErrorString(status); \
  } while (0)

template <typename T>
class CuDNNTraits;

template <>
class CuDNNTraits<float16> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
  static float oneval, zeroval;
  static const void *one, *zero;
};

template <>
class CuDNNTraits<bfloat16> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_BFLOAT16;
  static float oneval, zeroval;
  static const void *one, *zero;
};

template <>
class CuDNNTraits<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};

template <>
class CuDNNTraits<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

/*! \brief Return the cudnn data type by type */
DRAGON_API const cudnnDataType_t& CuDNNGetDataType(const TypeMeta& type);

/*! \brief Return the cudnn data type by template type */
template <typename T>
const cudnnDataType_t& CuDNNGetDataType() {
  return CuDNNGetDataType(TypeMeta::Make<T>());
}

/*! \brief Return the cudnn math type by template type */
template <typename T>
DRAGON_API cudnnMathType_t CuDNNGetMathType();

/*! \brief Create a tensor desc */
DRAGON_API void CuDNNCreateTensorDesc(cudnnTensorDescriptor_t* desc);

/*! \brief Destroy a tensor desc */
DRAGON_API void CuDNNDestroyTensorDesc(cudnnTensorDescriptor_t desc);

/*! \brief Set a tensor desc with dims */
template <typename T>
DRAGON_API void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t desc,
    const vec64_t& dims);

/*! \brief Set a tensor desc with dims and strides */
template <typename T>
DRAGON_API void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t desc,
    const vec64_t& dims,
    const vec64_t& strides);

/*! \brief Set a tensor desc with dims, data format and group */
template <typename T>
DRAGON_API void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t desc,
    const vec64_t& dims,
    const string& data_format);

/*! \brief Set a bias desc with expanding dimensions */
template <typename T>
DRAGON_API void CuDNNSetBiasDesc(
    cudnnTensorDescriptor_t desc,
    const int num_dims,
    const int64_t N,
    const string& data_format);

/*! \brief Set a dropout desc */
template <class Context>
DRAGON_API void CuDNNSetDropoutDesc(
    cudnnDropoutDescriptor_t desc,
    const float ratio,
    Context* ctx);

class DRAGON_API CuDNNTensorDescs {
 public:
  CuDNNTensorDescs(int num_descs) {
    descs_.resize(num_descs);
    for (int i = 0; i < num_descs; ++i) {
      CuDNNCreateTensorDesc(&descs_[i]);
    }
  }

  ~CuDNNTensorDescs() {
    for (auto desc : descs_) {
      CuDNNDestroyTensorDesc(desc);
    }
  }

  template <typename T>
  void Set(const vec64_t& dims, const vec64_t& strides) {
    CHECK_EQ(dims.size(), strides.size());
    for (auto desc : descs_) {
      CuDNNSetTensorDesc<T>(desc, dims, strides);
    }
  }

  cudnnTensorDescriptor_t* data() {
    return descs_.data();
  }

 protected:
  vector<cudnnTensorDescriptor_t> descs_;
};

} // namespace dragon

#endif // USE_CUDNN

#endif // DRAGON_UTILS_DEVICE_COMMON_CUDNN_H_
