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
class CuDNNType;

template <>
class CuDNNType<float16> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
  static float oneval, zeroval;
  static const void *one, *zero;
  typedef float BNParamType;
};

template <>
class CuDNNType<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
  typedef float BNParamType;
};

template <>
class CuDNNType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
  typedef double BNParamType;
};

/*! \brief Return the cudnn math type by template type */
template <typename T>
cudnnMathType_t CuDNNGetMathType();

/*! \brief Create a tensor desc */
void CuDNNCreateTensorDesc(cudnnTensorDescriptor_t* desc);

/*! \brief Destroy a tensor desc */
void CuDNNDestroyTensorDesc(cudnnTensorDescriptor_t desc);

/*! \brief Set a tensor desc with dims */
template <typename T>
void CuDNNSetTensorDesc(cudnnTensorDescriptor_t desc, const vec64_t& dims);

/*! \brief Set a tensor desc with dims and strides */
template <typename T>
void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t desc,
    const vec64_t& dims,
    const vec64_t& strides);

/*! \brief Set a tensor desc with dims, data format and group */
template <typename T>
void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t desc,
    const vec64_t& dims,
    const string& data_format);

/*! \brief Set a bias desc with expanding dimensions */
template <typename T>
void CuDNNSetBiasDesc(
    cudnnTensorDescriptor_t desc,
    const int num_dims,
    const int64_t N,
    const string& data_format);

/*! \brief Set a dropout desc */
template <class Context>
void CuDNNSetDropoutDesc(
    cudnnDropoutDescriptor_t desc,
    const float ratio,
    Context* ctx);

class CuDNNTensorDescs {
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
