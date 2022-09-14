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

#ifndef DRAGON_UTILS_DEVICE_COMMON_MPS_H_
#define DRAGON_UTILS_DEVICE_COMMON_MPS_H_

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
typedef NSDictionary* NSDictionary_t;
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLBuffer> MTLBuffer_t;
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
typedef MTLFunctionConstantValues* MTLFunctionConstantValues_t;
typedef MTLDataType MTLDataType_t;
typedef MPSCommandBuffer* MPSCommandBuffer_t;
typedef MPSShape* MPSShape_t;
typedef MPSDataType MPSDataType_t;
typedef MPSGraph* MPSGraph_t;
typedef MPSGraphTensor* MPSGraphTensor_t;
typedef MPSGraphTensorData* MPSGraphTensorData_t;
typedef MPSGraphExecutionDescriptor* MPSGraphExecutionDescriptor_t;
#else
struct NSDictionary;
struct MTLDevice;
struct MTLBuffer;
struct MTLLibrary;
struct MTLCommandQueue;
struct MTLComputePipelineState;
struct MTLComputeCommandEncoder;
struct MTLFunctionConstantValues;
struct MPSCommandBuffer;
struct MPSShape;
struct MPSGraph;
struct MPSGraphTensor;
struct MPSGraphTensorData;
struct MPSGraphExecutionDescriptor;
typedef NSDictionary* NSDictionary_t;
typedef MTLDevice* MTLDevice_t;
typedef MTLBuffer* MTLBuffer_t;
typedef MTLLibrary* MTLLibrary_t;
typedef MTLCommandQueue* MTLCommandQueue_t;
typedef unsigned long MTLDataType_t;
typedef MTLComputePipelineState* MTLComputePipelineState_t;
typedef MTLComputeCommandEncoder* MTLComputeCommandEncoder_t;
typedef MTLFunctionConstantValues* MTLFunctionConstantValues_t;
typedef MPSCommandBuffer* MPSCommandBuffer_t;
typedef MPSShape* MPSShape_t;
typedef uint32_t MPSDataType_t;
typedef MPSGraph* MPSGraph_t;
typedef MPSGraphTensor* MPSGraphTensor_t;
typedef MPSGraphTensorData* MPSGraphTensorData_t;
typedef MPSGraphExecutionDescriptor* MPSGraphExecutionDescriptor_t;
#define nil NULL
#endif

#include "dragon/core/common.h"

namespace dragon {

#ifdef USE_MPS

/*
 * Constants.
 */

/*! \brief The maximum number of devices in a single machine */
constexpr int MPS_MAX_DEVICES = 16;

/*! \brief The maximum number of tensor dimsensions */
constexpr int MPS_TENSOR_MAX_DIMS = 8;

/*
 * Defines.
 */

#define MPS_TENSOR_DIMS_CHECK(num_dims)        \
  CHECK_LE(num_dims, MPS_TENSOR_MAX_DIMS)      \
      << "Too many (> " << MPS_TENSOR_MAX_DIMS \
      << ") dimensions to launch the mps kernel."

/*
 * Classes.
 */

class MPSStream;

class MPSConstant {
 public:
  /*! \brief Constructor with a single index */
  MPSConstant(const void* value, MTLDataType_t data_type, int index = 0)
      : value_(value), data_type_(data_type), indices_({index}) {
    SetItemsize();
  }

  /*! \brief Constructor with a set of indices */
  MPSConstant(const void* value, MTLDataType_t data_type, vector<int> indices)
      : value_(value), data_type_(data_type), indices_(indices) {
    SetItemsize();
  }

  /*! \brief Return a string formatting this constant */
  const string ToString() const;

  /*! \brief Set this constant for the MTLFunction */
  void SetFor(MTLFunctionConstantValues_t values) const;

 private:
  /*! \brief Set the itemsize */
  void SetItemsize();

  /*! \brief Return a string formatting the value pointer */
  const string ToString(const void* value) const;

  /*! \brief The value pointer */
  const void* value_;

  /*! \brief The data type */
  MTLDataType_t data_type_;

  /*! \brief The itemsize */
  size_t itemsize_;

  /*! \brief The indices to set for the function */
  vector<int> indices_;
};

/*!
 * \brief The MPSGraphCache.
 */
class DRAGON_API MPSGraphCache {
 public:
  /*! \brief Return the cached graph placeholders */
  const vector<MPSGraphTensor_t>& GetPlaceholders(
      const vector<const vector<int64_t>*>& shapes,
      const vector<const TypeMeta*>& types,
      std::function<void(vector<MPSGraphTensor_t>&)> init_func) {
    int64_t key = 0;
    for (int64_t i = 0; i < int64_t(shapes.size()); ++i) {
      for (auto dim : *(shapes[i])) {
        key ^= hash_func_(dim) + 0x9e3779b9 + (key << 6) + (key >> 2) + i;
      }
    }
    for (auto* m : types) {
      key ^= hash_func_(m->id()) + 0x9e3779b9 + (key << 6) + (key >> 2) + 1024;
    }
    auto find_iter = map_.find(key);
    if (find_iter == map_.end()) {
      auto& placeholders = map_[key];
      init_func(placeholders);
      return placeholders;
    }
    return find_iter->second;
  }

  ~MPSGraphCache();

 private:
  /*! \brief The hash function */
  std::hash<int64_t> hash_func_;

  /*! \brief The graph placeholders of all shapes */
  Map<int64_t, vector<MPSGraphTensor_t>> map_;
};

/*
 * NS Functions.
 */

/*! \brief Return the retain count of a NSObject */
int NSGetRetainCount(const void* obj);

/*! \brief Call the retain metohd once for a NSObject */
void NSRetainObject(const void* obj);

/*! \brief Call the release metohd once for a NSObject */
void NSReleaseObject(const void* obj);

/*
 * Metal Functions.
 */

/*! \brief Return the number of available Metal devices */
int MTLGetDeviceCount();

/*! \brief Return the contents data of a MTLBuffer */
void* MTLGetBufferContents(const void* obj);

/*
 * MPS Functions.
 */

/*! \brief Return a MPSShape object from the dimensions */
MPSShape_t MPSGetShape(const vec64_t& dims);

/*! \brief Return the MPSDataType value by type */
MPSDataType_t MPSGetDataType(const TypeMeta& type);

/*! \brief Return the MPSDataType value by template type */
template <typename T>
MPSDataType_t MPSGetDataType() {
  return MPSGetDataType(TypeMeta::Make<T>());
}

/*! \brief Return the threads for block reduce kernel */
int MPSGetBlockReduceThreads(const int M, MTLComputePipelineState_t state);

/*! \brief Dispatch threads for 1D kernel */
void MPSDispatchThreads(
    const int N,
    MTLComputeCommandEncoder_t encoder,
    MTLComputePipelineState_t state);

/*! \brief Dispatch threads for 2D kernel */
void MPSDispatchThreads(
    const int N,
    const int M,
    MTLComputeCommandEncoder_t encoder,
    MTLComputePipelineState_t state);

/*
 * MPSGraph Functions.
 */

/*! \brief Create a MPSGraph object */
MPSGraph_t MPSCreateGraph();

/*! \brief Create a MPSGraphTensor object */
MPSGraphTensor_t MPSCreateTensor(
    MPSGraph_t graph,
    const vec64_t& dims,
    const TypeMeta& data_type);

/*! \brief Create a MPSGraphTensor object by template type */
template <typename T>
MPSGraphTensor_t MPSCreateTensor(MPSGraph_t graph, const vec64_t& dims) {
  return MPSCreateTensor(graph, dims, TypeMeta::Make<T>());
}

/*! \brief Create a MPSGraphTensorData object by buffer and tensor */
MPSGraphTensorData_t MPSCreateTensorData(
    MTLBuffer_t buffer,
    MPSGraphTensor_t tensor);

/*! \brief Create a MPSGraphTensorData object by data and tensor */
template <typename T>
MPSGraphTensorData_t MPSCreateTensorData(T* data, MPSGraphTensor_t tensor) {
  return MPSCreateTensorData(MTLBuffer_t(data), tensor);
}

#else

#define MPS_NOT_COMPILED LOG(FATAL) << "MPS library is not built with."

#endif

} // namespace dragon

#endif // DRAGON_UTILS_DEVICE_COMMON_MPS_H_
