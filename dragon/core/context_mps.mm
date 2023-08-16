#include "dragon/core/context_mps.h"
#include "dragon/core/workspace.h"

namespace dragon {

MPSObjects::MPSObjects() {
  devices_.clear();
  auto* devices = MTLCopyAllDevices();
  for (int i = 0; i < [devices count]; ++i) {
    auto* device = devices[i];
    if (!device.isLowPower) {
      devices_.push_back([device retain]);
      random_seeds_.push_back(DEFAULT_RNG_SEED);
    }
  }
}

MPSObjects::~MPSObjects() {
  for (int device_id = 0; device_id < devices_.size(); ++device_id) {
    for (auto& iter1 : states_[device_id]) {
      for (auto& iter2 : iter1.second) {
        [iter2.second release];
      }
    }
    for (auto& iter : libraries_[device_id]) {
      [iter.second release];
    }
    for (auto* stream : streams_[device_id]) {
      if (stream) delete stream;
    }
    [devices_[device_id] release];
    devices_[device_id] = nil;
    for (auto& workspace : workspaces_[device_id]) {
      if (workspace) delete workspace;
    }
  }
}

string MPSObjects::GetDeviceName(int device_id) {
  if (device_id >= devices_.size()) return "";
  const auto device = devices_[device_id];
  return [[device name] UTF8String];
}

vector<string> MPSObjects::GetDeviceFamily(int device_id) {
  if (device_id >= devices_.size()) return {};
  vector<string> ret;
  const auto device = devices_[device_id];
  if ([device supportsFamily:MTLGPUFamilyMac2]) {
    ret.push_back("Mac2"); // AMD, Apple
  }
  if ([device supportsFamily:MTLGPUFamilyApple6]) {
    ret.push_back("Apple6"); // A13
  }
  if ([device supportsFamily:MTLGPUFamilyApple7]) {
    ret.push_back("Apple7"); // A14, M1
  }
#if (MPS_OSX_VERSION_MAJOR >= 13)
  if ([device supportsFamily:MTLGPUFamilyApple8]) {
    ret.push_back("Apple8"); // A15, M2
  }
#endif
  return ret;
}

MPSDeviceGuard::MPSDeviceGuard(int new_id) {
  prev_id_ = MPSContext::current_device();
  if (prev_id_ != new_id) {
    MPSContext::objects().SetDevice(new_id);
  }
}

MPSDeviceGuard::~MPSDeviceGuard() {
  MPSContext::objects().SetDevice(prev_id_);
}

MPSStream::MPSStream(int device_id) : device_id_(device_id) {
  MPSDeviceGuard guard(device_id);
  auto* device = MPSContext::objects().devices_[device_id];
  command_queue_ = [device newCommandQueue];
  execution_desc_ = [MPSGraphExecutionDescriptor new];
}

MPSStream::~MPSStream() {
  [command_queue_ release];
  [execution_desc_ release];
  if (command_buffer_) CommitAndWait();
}

void MPSStream::Commit() {
  [command_buffer() commitAndContinue];
}

void MPSStream::CommitAndWait() {
  [command_buffer() commit];
  [command_buffer_ waitUntilCompleted];
  [command_buffer_ release]; // retainCount: 1
  command_buffer_ = nil; // retainCount: 0
}

void MPSStream::Encode(
    MPSGraph_t graph,
    NSDictionary_t inputs,
    NSDictionary_t outputs) {
  [graph encodeToCommandBuffer:command_buffer()
                         feeds:inputs
              targetOperations:nil
             resultsDictionary:outputs
           executionDescriptor:execution_desc_];
}

void MPSStream::CreatePhiloxState(MPSGraph_t graph, uint32_t seed, int* state) {
  @autoreleasepool {
    auto* placeholder = [graph randomPhiloxStateTensorWithSeed:seed name:nil];
    auto* outputs = @{placeholder : MPSCreateTensorData(state, placeholder)};
    Encode(graph, @{}, outputs);
  }
  CommitAndWait();
}

MPSCommandBuffer_t MPSStream::command_buffer() {
  if (!command_buffer_) {
    command_buffer_ =
        [MPSCommandBuffer commandBufferFromCommandQueue:command_queue_];
    // Command buffer is created as autoreleased.
    // Retain it to work with autoreleasepool.
    [command_buffer_ retain];
  }
  return command_buffer_;
}

void* MPSContext::New(size_t size) {
  auto* device = objects().devices_[current_device()];
  auto* data = [device newBufferWithLength:size
                                   options:MTLResourceStorageModePrivate];
  if (!data) {
    LOG(FATAL) << "\nAllocate device buffer with " << size << " bytes failed.";
  }
  return data;
}

void* MPSContext::NewShared(size_t size) {
  auto* device = objects().devices_[current_device()];
  auto* data = [device newBufferWithLength:size
                                   options:[device hasUnifiedMemory]
                                       ? MTLResourceStorageModeShared
                                       : MTLResourceStorageModeManaged];
  if (!data) {
    LOG(FATAL) << "\nAllocate shared buffer with " << size << " bytes failed.";
  }
  return data;
}

void* MPSContext::NewSharedFromBytes(size_t size, const void* src) {
  auto* device = objects().devices_[current_device()];
  auto* data = [device newBufferWithBytes:src
                                   length:size
                                  options:[device hasUnifiedMemory]
                                      ? MTLResourceStorageModeShared
                                      : MTLResourceStorageModeManaged];
  if (!data) {
    LOG(FATAL) << "\nAllocate shared buffer with " << size << " bytes failed.";
  }
  return data;
}

void* MPSContext::NewSharedFromBuffer(const void* src) {
  auto* device = objects().devices_[current_device()];
  auto* src_data = id<MTLBuffer>(src);
  auto* dest_data = [device newBufferWithLength:src_data.length
                                        options:MTLResourceStorageModeShared];
  if (!dest_data) {
    LOG(FATAL) << "\nAllocate shared buffer with " << src_data.length
               << " bytes failed.";
  }
  Memcpy<MPSContext, MPSContext>(dest_data.length, dest_data, src_data);
  return dest_data;
}

void MPSContext::Delete(void* ptr) {
  [id<MTLBuffer>(ptr) release];
}

void MPSContext::Memset(size_t n, void* ptr, int value) {
  auto* stream = objects().default_stream();
  @autoreleasepool {
    auto* encoder = [stream->command_buffer() blitCommandEncoder];
    [encoder fillBuffer:id<MTLBuffer>(ptr) range:NSMakeRange(0, n) value:value];
    [encoder endEncoding];
  }
  SynchronizeStream(stream);
}

void MPSContext::MemsetAsync(size_t n, void* ptr, int value) {
  auto* encoder = [mps_stream()->command_buffer() blitCommandEncoder];
  [encoder fillBuffer:id<MTLBuffer>(ptr) range:NSMakeRange(0, n) value:value];
  [encoder endEncoding];
  [encoder release];
}

template <>
void MPSContext::Memcpy<MPSContext, MPSContext>(
    size_t n,
    void* dest,
    const void* src,
    int device) {
  auto* stream = objects().default_stream(device);
  auto* encoder = [stream->command_buffer() blitCommandEncoder];
  [encoder copyFromBuffer:id<MTLBuffer>(src)
             sourceOffset:0
                 toBuffer:id<MTLBuffer>(dest)
        destinationOffset:0
                     size:n];
  [encoder endEncoding];
  [encoder release];
  SynchronizeStream(stream);
}

void MPSContext::SynchronizeResource(MPSStream* stream, size_t n, void* ptr) {
  if (stream == nullptr) {
    [id<MTLBuffer>(ptr) didModifyRange:NSMakeRange(0, n)];
  } else {
    if (stream->command_queue().device.hasUnifiedMemory) return;
    auto* encoder = [stream->command_buffer() blitCommandEncoder];
    [encoder synchronizeResource:id<MTLResource>(ptr)];
    [encoder endEncoding];
    [encoder release];
  }
}

template <>
void MPSContext::MemcpyAsync<MPSContext, MPSContext>(
    size_t n,
    void* dest,
    const void* src) {
  auto* encoder = [mps_stream()->command_buffer() blitCommandEncoder];
  [encoder copyFromBuffer:id<MTLBuffer>(src)
             sourceOffset:0
                 toBuffer:id<MTLBuffer>(dest)
        destinationOffset:0
                     size:n];
  [encoder endEncoding];
  [encoder release];
}

MTLComputePipelineState_t MPSKernel::GetState(
    MPSContext* ctx,
    const vector<MPSConstant>& constants) {
  // Find the created pipeline state.
  string kernel_key = kernel_name_;
  for (size_t i = 0; i < constants.size(); ++i) {
    kernel_key += "_" + constants[i].ToString();
  }
  auto& objects = ctx->objects();
  auto& states = objects.states_[ctx->device()][kernel_name_];
  auto find_state_iter = states.find(kernel_key);
  if (find_state_iter != states.end()) return find_state_iter->second;

  // Find the compiled library.
  MTLLibrary_t lib = nil;
  auto* device = objects.devices_[ctx->device()];
  auto& libraries = objects.libraries_[ctx->device()];
  auto find_lib_iter = libraries.find(lib_key_);
  if (find_lib_iter != libraries.end()) {
    lib = find_lib_iter->second;
  } else {
    @autoreleasepool {
      NSError* error = nil;
      auto* source = [NSString stringWithUTF8String:lib_source_.c_str()];
      auto* options = [[MTLCompileOptions alloc] init];
      // Metal 2.0: function constant
      // Metal 2.2: hostname attribute
      // Metal 2.3: int64 buffer
      // Metal 3.0: float32 atomic
      // Metal 3.1: bfloat16
#if (MPS_OSX_VERSION_MAJOR >= 14)
      [options setLanguageVersion:MTLLanguageVersion3_1];
#elif (MPS_OSX_VERSION_MAJOR >= 13)
      [options setLanguageVersion:MTLLanguageVersion3_0];
#else
      [options setLanguageVersion:MTLLanguageVersion2_3];
#endif
      [options setFastMathEnabled:YES];
      lib = [device newLibraryWithSource:source options:options error:&error];
      if (error) LOG(FATAL) << error.localizedDescription.UTF8String;
      libraries[lib_key_] = lib;
      [options release];
    }
  }

  // Set the constant values.
  auto* constant_values = [MTLFunctionConstantValues new];
  for (size_t i = 0; i < constants.size(); ++i) {
    constants[i].SetFor(constant_values);
  }

  // Create the pipeline state.
  MTLComputePipelineState_t state = nil;
  @autoreleasepool {
    NSError* error = nil;
    auto* name = [NSString stringWithUTF8String:kernel_name_.c_str()];
    auto* func = [lib newFunctionWithName:name
                           constantValues:constant_values
                                    error:&error];
    if (func == nil) LOG(FATAL) << error.localizedDescription.UTF8String;
    state = [device newComputePipelineStateWithFunction:func error:&error];
    if (state == nil) LOG(FATAL) << error.localizedDescription.UTF8String;
    states[kernel_key] = state;
  }
  [constant_values release];
  return state;
}

Workspace* MPSObjects::workspace(int device_id, int stream_id) {
  auto& workspaces = workspaces_[device_id];
  if (workspaces.size() <= unsigned(stream_id)) {
    workspaces.resize(stream_id + 1, nullptr);
  }
  if (!workspaces[stream_id]) {
    workspaces[stream_id] = new Workspace("");
    workspaces[stream_id]
        ->CreateTensor("MPSPhiloxStateInc")
        ->template CopyFrom<int>(vec32_t({1, 1, 1, 1, 0, 0, 0}));
  }
  return workspaces[stream_id];
}

std::mutex& MPSContext::mutex() {
  static std::mutex m;
  return m;
}

MPSObjects& MPSContext::objects() {
  static thread_local MPSObjects objects_;
  return objects_;
}

} // namespace dragon
