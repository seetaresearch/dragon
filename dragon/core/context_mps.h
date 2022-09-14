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

#ifndef DRAGON_CORE_CONTEXT_MPS_H_
#define DRAGON_CORE_CONTEXT_MPS_H_

#include "dragon/core/common.h"
#include "dragon/utils/device/common_mps.h"

namespace dragon {

#ifdef USE_MPS

class Workspace;

/*!
 * \brief The MPSDevideGuard.
 */
class DRAGON_API MPSDeviceGuard {
 public:
  /*! \brief Constructor */
  explicit MPSDeviceGuard(int new_id);

  /*! \brief Destructor */
  ~MPSDeviceGuard();

 private:
  int prev_id_;
};

/*!
 * \brief The MPSStream.
 */
class DRAGON_API MPSStream {
 public:
  /*! \brief Constructor */
  explicit MPSStream(int device_id);

  /*! \brief Destructor */
  ~MPSStream();

  /*! \brief Commit commands */
  void Commit();

  /*! \brief Commit commands and wait until completed */
  void CommitAndWait();

  /*! \brief Encode the commands of given mps graph */
  void Encode(MPSGraph_t graph, NSDictionary_t inputs, NSDictionary_t outputs);

  MTLCommandQueue_t command_queue() {
    return command_queue_;
  }

  MPSCommandBuffer_t command_buffer();

 private:
  int device_id_;
  MTLCommandQueue_t command_queue_ = nil;
  MPSCommandBuffer_t command_buffer_ = nil;
  MPSGraphExecutionDescriptor_t execution_desc_ = nil;
};

/*!
 * \brief The MPSObjects.
 */
class MPSObjects {
 public:
  /*! \brief Constructor */
  MPSObjects();

  /*! \brief Destructor */
  ~MPSObjects();

  /*! \brief Set the current mps device */
  void SetDevice(int device_id) {
    if (device_id >= int(devices_.size())) {
      LOG(FATAL) << "MPS error: invalid device ordinal " << device_id;
    }
    device_id_ = device_id;
  }

  /*! \brief Return the current mps device */
  int GetDevice() {
    return device_id_;
  }

  /*! \brief Return the name of given device */
  string GetDeviceName(int device_id);

  /*! \brief Return the family set that given device supports */
  vector<string> GetDeviceFamily(int device_id);

  /*! \brief Return the default mps stream of current device */
  MPSStream* default_stream() {
    return stream(device_id_, 0);
  }

  /*! \brief Return the default mps stream of given device */
  MPSStream* default_stream(int device_id) {
    return stream(device_id, 0);
  }

  /*! \brief Return the specified mps stream */
  MPSStream* stream(int device_id, int stream_id) {
    auto& streams = streams_[device_id];
    if (streams.size() <= unsigned(stream_id)) {
      streams.resize(stream_id + 1, nullptr);
    }
    if (!streams[stream_id]) {
      streams[stream_id] = new MPSStream(device_id);
    }
    return streams[stream_id];
  }

  /*! \brief Return the workspace of specified mps stream */
  Workspace* workspace(int device_id, int stream_id);

  /*! \brief The current mps device */
  int device_id_ = 0;

  /*! \brief The retained mps devices */
  vector<MTLDevice_t> devices_;

  /*! \brief The created mps streams for all devices */
  vector<MPSStream*> streams_[MPS_MAX_DEVICES];

  /*! \brief The created mps workspaces for all devices */
  vector<Workspace*> workspaces_[MPS_MAX_DEVICES];

  /*! \brief The compiled mps libraries for all devices */
  Map<string, MTLLibrary_t> libraries_[MPS_MAX_DEVICES];

  /*! \brief The create mps pipeline states for all devices */
  Map<string, Map<string, MTLComputePipelineState_t>> states_[MPS_MAX_DEVICES];

 private:
  DISABLE_COPY_AND_ASSIGN(MPSObjects);
};

/*!
 * \brief The MPS device context.
 */
class DRAGON_API MPSContext {
 public:
  /*! \brief Constructor */
  MPSContext() : device_id_(0), random_seed_(DEFAULT_RNG_SEED) {}

  /*! \brief Constructor with the device index */
  explicit MPSContext(int device) : device_id_(device) {}

  /*! \brief Constructor with the device option */
  explicit MPSContext(const DeviceOption& option)
      : device_id_(option.device_id()),
        random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : DEFAULT_RNG_SEED) {
    CHECK_EQ(option.device_type(), PROTO_MPS);
  }

  /*! \brief Allocate a device buffer */
  static void* New(size_t size);

  /*! \brief Allocate a shared buffer */
  static void* NewShared(size_t size);

  /*! \brief Allocate a shared buffer from given bytes data */
  static void* NewSharedFromBytes(size_t size, const void* src);

  /*! \brief Allocate a shared buffer from the given buffer */
  static void* NewSharedFromBuffer(const void* src);

  /*! \brief Deallocate a buffer */
  static void Delete(void* ptr);

  /*! \brief Set a buffer to the given value */
  static void Memset(size_t n, void* ptr, int value = 0);

  /*! \brief Copy a buffer to the destination */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t n, void* dest, const void* src) {
    Memcpy<DestContext, SrcContext>(n, dest, src, current_device());
  }

  /*! \brief Copy a bufferto the destination using given device */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t n, void* dest, const void* src, int device);

  /*! \brief Set a buffer to the given value asynchronously */
  void MemsetAsync(size_t n, void* ptr, int value = 0);

  /*! \brief Copy a buffer to the destination asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t n, void* dest, const void* src);

  /*! \brief Switch to the device and select given stream in current thread */
  void SwitchToDevice(int stream_id = 0) {
    objects().SetDevice(device_id_);
    stream_id_ = stream_id;
  }

  /*! \brief Synchronize the given stream */
  static void SynchronizeStream(MPSStream* stream) {
    stream->CommitAndWait();
  }

  /*! \brief Synchronize the given resource */
  static void SynchronizeResource(MPSStream* stream, size_t n, void* ptr);

  /*! \brief Wait for the dispatched computation to complete */
  void FinishDeviceComputation() {
    SynchronizeStream(mps_stream());
  }

  /*! \brief Return the current workspace */
  Workspace* workspace() {
    return objects().workspace(device_id_, stream_id_);
  }

  /*! \brief Return the specified workspace */
  Workspace* workspace(int device, int stream) {
    return objects().workspace(device, stream);
  }

  /*! \brief Return the current mps stream */
  MPSStream* mps_stream() {
    return objects().stream(device_id_, stream_id_);
  }

  /*! \brief Return the specified mps stream */
  MPSStream* mps_stream(int device, int stream) {
    return objects().stream(device, stream);
  }

  /*! \brief Return the device index */
  int device() const {
    return device_id_;
  }

  /*! \brief Return the stream index */
  int stream() const {
    return stream_id_;
  }

  /*! \brief Return the device index of current thread */
  static int current_device() {
    return objects().GetDevice();
  }

  /*! \brief Return the shared context mutex */
  static std::mutex& mutex();

  /*! \brief Return the thread-local mps objects */
  static MPSObjects& objects();

  /*! \brief Return the random generator */
  std::mt19937* rand_generator() {
    if (!rand_generator_.get()) {
      rand_generator_.reset(new std::mt19937(random_seed_));
    }
    return rand_generator_.get();
  }

  /*! \brief Set the stream index */
  void set_stream(int stream) {
    stream_id_ = stream;
  }

 private:
  int device_id_, stream_id_ = 0, random_seed_;
  unique_ptr<std::mt19937> rand_generator_;
};

/*!
 * \brief The MPSKernel.
 */
class DRAGON_API MPSKernel {
 public:
  /*! \brief Constructor */
  MPSKernel(const string& kernel_name, const string& lib_source)
      : kernel_name_(kernel_name), lib_source_(lib_source) {
    lib_key_ = std::to_string(intptr_t(&lib_source));
  }

  /*! \brief Return a typed kernel name */
  template <typename T>
  static string TypedString(const string& base) {
    static std::unordered_map<TypeId, string> m{
        {TypeMeta::Id<bool>(), "_bool"},
        {TypeMeta::Id<uint8_t>(), "_uint8_t"},
        {TypeMeta::Id<int8_t>(), "_int8_t"},
        {TypeMeta::Id<int>(), "_int"},
        {TypeMeta::Id<int64_t>(), "_int64_t"},
        {TypeMeta::Id<float16>(), "_half"},
        {TypeMeta::Id<float>(), "_float"},
        {TypeMeta::Id<double>(), "_double"},
    };
    auto it = m.find(TypeMeta::Make<T>().id());
    return it != m.end() ? base + it->second : base;
  }

  /*! \brief Return the pipeline state */
  MTLComputePipelineState_t GetState(MPSContext* ctx) {
    return GetState(ctx, {});
  }

  /*! \brief Return the pipeline state by constants */
  MTLComputePipelineState_t GetState(
      MPSContext* ctx,
      const vector<MPSConstant>&);

 private:
  /*! \brief The kernel name */
  string kernel_name_;

  /*! \brief The cache key of library where kernel defined */
  string lib_key_;

  /*! \brief The source code of library where kernel defined */
  const string& lib_source_;
};

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_CORE_CONTEXT_MPS_H_
