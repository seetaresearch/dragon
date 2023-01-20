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

#ifndef DRAGON_UTILS_DEVICE_COMMON_MLU_H_
#define DRAGON_UTILS_DEVICE_COMMON_MLU_H_

#ifdef USE_MLU
#include <cncl.h>
#include <cnnl.h>
#include <cnrt.h>
#endif

#include "dragon/core/common.h"

namespace dragon {

#ifdef USE_MLU

/*
 * Constants.
 */

/*! \brief The maximum number of devices in a single machine */
constexpr int MLU_MAX_DEVICES = 16;

/*! \brief The maximum number of tensor dimsensions */
constexpr int MLU_TENSOR_MAX_DIMS = 8;

/*
 * Defines.
 */

#ifdef CNRT_CHECK
#undef CNRT_CHECK
#endif

#ifdef CNCL_CHECK
#undef CNCL_CHECK
#endif

#define CNRT_CHECK(condition)                                   \
  do {                                                          \
    cnrtRet_t ret = condition;                                  \
    CHECK_EQ(ret, cnrtSuccess) << "\n" << cnrtGetErrorStr(ret); \
  } while (0)

#define CNNL_CHECK(condition)                                              \
  do {                                                                     \
    cnnlStatus_t ret = condition;                                          \
    CHECK_EQ(ret, CNNL_STATUS_SUCCESS) << "\n" << cnnlGetErrorString(ret); \
  } while (0)

#define CNCL_CHECK(condition)                                        \
  do {                                                               \
    cnclResult_t ret = condition;                                    \
    CHECK_EQ(ret, CNCL_RET_SUCCESS) << "\n" << cnclGetErrorStr(ret); \
  } while (0)

#define MLU_1D_KERNEL_LOOP(i, N, M) \
  for (int i = taskId * M; i < N; i += taskDim * M)

/*
 * CNRT Utilities.
 */

inline cnrtDim3_t MLU_BLOCKS() {
  int device, cluster_count, cores_per_cluster;
  CNRT_CHECK(cnrtGetDevice(&device));
  cnrtDeviceGetAttribute(&cluster_count, cnrtAttrClusterCount, device);
  cnrtDeviceGetAttribute(&cores_per_cluster, cnrtAttrMcorePerCluster, device);
  return cnrtDim3_t({uint32_t(cluster_count * cores_per_cluster), 1, 1});
}

inline cnrtDim3_t MLU_BLOCKS(const int N, const int M) {
  const cnrtDim3_t dim = MLU_BLOCKS();
  const unsigned int num_blocks = (N + M - 1) / M;
  return cnrtDim3_t({std::min(dim.x, num_blocks), 1, 1});
}

inline int MLUGetDeviceCount() {
  static int count = -1;
  if (count < 0) {
    unsigned int ret;
    auto err = cnrtGetDeviceCount(&ret);
    if (err == cnrtErrorNoDevice) {
      count = 0;
    } else {
      count = int(ret);
    }
  }
  return count;
}

inline int MLUGetDevice() {
  int device_id;
  CNRT_CHECK(cnrtGetDevice(&device_id));
  return device_id;
}

struct MLUDeviceProps {
  MLUDeviceProps() : props(MLUGetDeviceCount()) {
    for (int i = 0; i < props.size(); ++i) {
      CNRT_CHECK(cnrtGetDeviceProperties(&props[i], i));
    }
  }
  vector<cnrtDeviceProp_t> props;
};

inline const cnrtDeviceProp_t& MLUGetDeviceProp(int device_id) {
  static MLUDeviceProps props;
  CHECK_LT(device_id, int(props.props.size()))
      << "\nInvalid device id: " << device_id << "\nFound "
      << props.props.size() << " devices.";
  return props.props[device_id];
}

class MLUDeviceGuard {
 public:
  explicit MLUDeviceGuard(int new_id) {
    CNRT_CHECK(cnrtGetDevice(&prev_id_));
    if (prev_id_ != new_id) {
      CNRT_CHECK(cnrtSetDevice(new_id));
    }
  }

  ~MLUDeviceGuard() {
    CNRT_CHECK(cnrtSetDevice(prev_id_));
  }

 private:
  int prev_id_;
};

/*
 * CNNL Utilities.
 */

template <typename T>
class CNNLType;

template <>
class CNNLType<float16> {
 public:
  static const cnnlDataType_t type = CNNL_DTYPE_HALF;
  static float oneval, zeroval;
  static const void *one, *zero;
  typedef float BNParamType;
};

template <>
class CNNLType<float> {
 public:
  static const cnnlDataType_t type = CNNL_DTYPE_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
  typedef float BNParamType;
};

template <>
class CNNLType<double> {
 public:
  static const cnnlDataType_t type = CNNL_DTYPE_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
  typedef double BNParamType;
};

/*! \brief Return the cnnl data type by type */
const cnnlDataType_t& CNNLGetDataType(const TypeMeta& type);

/*! \brief Return the cnnl data type by template type */
template <typename T>
const cnnlDataType_t& CNNLGetDataType() {
  return CNNLGetDataType(TypeMeta::Make<T>());
}

/*! \brief Create a tensor desc */
void CNNLCreateTensorDesc(cnnlTensorDescriptor_t* desc);

/*! \brief Destroy a tensor desc */
void CNNLDestroyTensorDesc(cnnlTensorDescriptor_t desc);

/*! \brief Set a tensor desc with dims */
template <typename T>
void CNNLSetTensorDesc(cnnlTensorDescriptor_t desc, const vec64_t& dims);

/*! \brief Set a tensor desc with dims and strides */
template <typename T>
void CNNLSetTensorDesc(
    cnnlTensorDescriptor_t desc,
    const vec64_t& dims,
    const vec64_t& strides);

/*! \brief Set a tensor desc with dims, data format */
template <typename T>
void CNNLSetTensorDesc(
    cnnlTensorDescriptor_t desc,
    const vec64_t& dims,
    const string& data_format);

#else

#define MLU_NOT_COMPILED LOG(FATAL) << "MLU library is not built with."

class MLUDeviceGuard {
 public:
  explicit MLUDeviceGuard(int new_id) {
    MLU_NOT_COMPILED;
  }
};

#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_UTILS_DEVICE_COMMON_MLU_H_
