#include "dragon/utils/device/common_mps.h"

namespace dragon {

int NSGetRetainCount(const void* obj) {
  return [id<NSObject>(obj) retainCount];
}

void NSRetainObject(const void* obj) {
  [id<NSObject>(obj) retain];
}

void NSReleaseObject(const void* obj) {
  [id<NSObject>(obj) release];
}

int MTLGetDeviceCount() {
  static int count = -1;
  if (count < 0) {
    count = 0;
    @autoreleasepool {
      auto* devices = [MTLCopyAllDevices() autorelease];
      for (int i = 0; i < [devices count]; ++i) {
        if (![devices[i] isLowPower]) count++;
      }
    }
  }
  return count;
}

void* MTLGetBufferContents(const void* obj) {
  return [id<MTLBuffer>(obj) contents];
}

void MPSConstant::SetItemsize() {
  if (data_type_ == MTLDataTypeUInt) {
    itemsize_ = 4;
  } else if (data_type_ == MTLDataTypeFloat) {
    itemsize_ = 4;
  } else if (data_type_ == MTLDataTypeInt) {
    itemsize_ = 4;
  } else if (data_type_ == MTLDataTypeBool) {
    itemsize_ = 1;
  } else if (data_type_ == MTLDataTypeUInt4) {
    itemsize_ = 16;
  } else {
    LOG(FATAL) << "Unsupported MTLDataType: " << data_type_;
  }
}

const string MPSConstant::ToString(const void* value) const {
  if (data_type_ == MTLDataTypeUInt) {
    return str::to(((uint32_t*)value)[0]);
  } else if (data_type_ == MTLDataTypeFloat) {
    return str::to(((float*)value)[0]);
  } else if (data_type_ == MTLDataTypeInt) {
    return str::to(((int*)value)[0]);
  } else if (data_type_ == MTLDataTypeBool) {
    return str::to(((bool*)value)[0]);
  } else if (data_type_ == MTLDataTypeUInt4) {
    string ret = "(" + str::to(((uint32_t*)value)[0]) + ",";
    ret += str::to(((uint32_t*)value)[1]) + ",";
    ret += str::to(((uint32_t*)value)[2]) + ",";
    ret += str::to(((uint32_t*)value)[3]) + ")";
    return ret;
  }
  return "";
}

const string MPSConstant::ToString() const {
  if (indices_.size() == 1) return ToString(value_);
  string ret = ToString(value_);
  const auto* value = (const uint8_t*)value_ + itemsize_;
  for (size_t i = 1; i < indices_.size(); ++i, value += itemsize_) {
    ret += ToString(value);
  }
  return ret;
}

void MPSConstant::SetFor(MTLFunctionConstantValues_t values) const {
  const auto* value = (const uint8_t*)value_;
  for (size_t i = 0; i < indices_.size(); ++i, value += itemsize_) {
    [values setConstantValue:value type:data_type_ atIndex:indices_[i]];
  }
}

MPSShape_t MPSGetShape(const vec64_t& dims) {
  if (dims.empty()) return nil;
  const int num_dims = dims.size();
  NSNumber* shape[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    shape[i] = [NSNumber numberWithInteger:dims[i]];
  }
  return [NSArray arrayWithObjects:shape count:num_dims];
}

MPSDataType_t MPSGetDataType(const TypeMeta& type) {
  static MPSDataType unknown_type = MPSDataTypeInvalid;
  static std::unordered_map<TypeId, MPSDataType> m {
    {TypeMeta::Id<bool>(), MPSDataTypeBool}, // macOS 12.0 or higher.
        {TypeMeta::Id<uint8_t>(), MPSDataTypeUInt8},
        {TypeMeta::Id<int8_t>(), MPSDataTypeInt8},
        {TypeMeta::Id<int>(), MPSDataTypeInt32},
        {TypeMeta::Id<int64_t>(), MPSDataTypeInt64},
        {TypeMeta::Id<float16>(), MPSDataTypeFloat16},
#if (MPS_OSX_VERSION_MAJOR >= 14)
        {TypeMeta::Id<bfloat16>(), MPSDataTypeBFloat16},
#endif
        {TypeMeta::Id<float>(), MPSDataTypeFloat32},
  };
  auto it = m.find(type.id());
  return it != m.end() ? it->second : unknown_type;
}

int MPSGetBlockReduceThreads(const int M, MTLComputePipelineState_t state) {
  const int warp_size = state.threadExecutionWidth; // 8, 16, 32, 64, ...
  const int max_threads = state.maxTotalThreadsPerThreadgroup;
  const int block_warps = std::min((M - 1) / warp_size + 1, warp_size);
  return std::min(block_warps * warp_size, max_threads);
}

void MPSDispatchThreads(
    const int N,
    MTLComputeCommandEncoder_t encoder,
    MTLComputePipelineState_t state) {
  const int max_threads = state.maxTotalThreadsPerThreadgroup;
  [encoder dispatchThreads:MTLSizeMake(N, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(std::min(N, max_threads), 1, 1)];
}

void MPSDispatchThreads(
    const int N,
    const int M,
    MTLComputeCommandEncoder_t encoder,
    MTLComputePipelineState_t state) {
  const int max_threads = state.maxTotalThreadsPerThreadgroup;
  [encoder dispatchThreadgroups:MTLSizeMake(N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(std::min(M, max_threads), 1, 1)];
}

MPSGraph_t MPSCreateGraph() {
  auto graph = [[MPSGraph new] autorelease];
  graph.options = MPSGraphOptionsNone;
  return graph;
}

MPSGraphTensor_t MPSCreateTensor(
    MPSGraph_t graph,
    const vec64_t& dims,
    const TypeMeta& data_type) {
  return [graph placeholderWithShape:MPSGetShape(dims)
                            dataType:MPSGetDataType(data_type)
                                name:nil];
}

MPSGraphTensorData_t MPSCreateTensorData(
    MTLBuffer_t buffer,
    MPSGraphTensor_t tensor) {
  return [[[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                  shape:tensor.shape
                                               dataType:tensor.dataType]
      autorelease];
}

MPSGraphCache::~MPSGraphCache() {
  for (auto iter : map_) {
    for (auto tensor : iter.second) {
      [tensor release];
    }
  }
}

} // namespace dragon
