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

#ifndef DRAGON_CORE_CONTEXT_MLU_H_
#define DRAGON_CORE_CONTEXT_MLU_H_

#include "dragon/core/common.h"
#include "dragon/utils/device/common_mlu.h"

namespace dragon {

#ifdef USE_MLU

class Workspace;

class DRAGON_API MLUObjects {
 public:
  /*! \brief Constructor */
  MLUObjects() {
    for (int i = 0; i < MLU_MAX_DEVICES; ++i) {
      random_seeds_[i] = DEFAULT_RNG_SEED;
      streams_[i] = vector<cnrtQueue_t>();
      workspaces_[i] = vector<Workspace*>();
      cnnl_handles_[i] = vector<cnnlHandle_t>();
      cnrand_generators_[i] = Map<string, cnnlRandGenerator_t>();
      cncl_comms_[i] = Map<string, cnclComm_t>();
    }
  }

  /*! \brief Destructor */
  ~MLUObjects();

  /*! \brief Set the current device */
  void SetDevice(int device_id) {
    CNRT_CHECK(cnrtSetDevice(device_id));
  }

  /*! \brief Set the random seed for given device */
  void SetRandomSeed(int device_id, int seed) {
    if (device_id < MLU_MAX_DEVICES) random_seeds_[device_id] = seed;
  }

  /*! \brief Return the current mlu device */
  int GetDevice() {
    int device_id;
    CNRT_CHECK(cnrtGetDevice(&device_id));
    return device_id;
  }

  /*! \brief Return the random seed of given device */
  int GetRandomSeed(int device_id) const {
    return (device_id < MLU_MAX_DEVICES) ? random_seeds_[device_id]
                                         : DEFAULT_RNG_SEED;
  }

  /*! \brief Return the specified mlu stream */
  cnrtQueue_t stream(int device_id, int stream_id);

  /*! \brief Return the specified cnnl handle */
  cnnlHandle_t cnnl_handle(int device_id, int stream_id);

  /*! \brief Return the specified cnrand generator */
  std::pair<cnnlRandGenerator_t, void*>
  cnrand_generator(int device_id, int stream_id, int seed);

  /*! \brief Return the specified cncl comm */
  cnclComm_t cncl_comm(
      int device_id,
      const string& cache_key,
      cnclCliqueId* comm_uuid,
      int comm_size,
      int comm_rank);

  /*! \brief Return the default mlu stream of current device */
  cnrtQueue_t default_stream() {
    return stream(GetDevice(), 0);
  }

  /*! \brief Return the default mlu stream of given device */
  cnrtQueue_t default_stream(int device_id) {
    return stream(device_id, 0);
  }

  /*! \brief Return the workspace of specified mlu stream */
  Workspace* workspace(int device_id, int stream_id);

  /*! \brief The random seed for all devices */
  int random_seeds_[MLU_MAX_DEVICES];

  /*! \brief The created streams for all devices */
  vector<cnrtQueue_t> streams_[MLU_MAX_DEVICES];

  /*! \brief The created workspaces for all devices */
  vector<Workspace*> workspaces_[MLU_MAX_DEVICES];

  /*! \brief The created cnnl handles for all devices */
  vector<cnnlHandle_t> cnnl_handles_[MLU_MAX_DEVICES];

  /*! \brief The created cnrand generators for all devices */
  Map<string, cnnlRandGenerator_t> cnrand_generators_[MLU_MAX_DEVICES];

  /*! \brief The created cncl comms for all devices */
  Map<string, cnclComm_t> cncl_comms_[MLU_MAX_DEVICES];

  /*! \brief The flag that uses CNNL or not */
  bool cnnl_enabled_ = true;

 private:
  DISABLE_COPY_AND_ASSIGN(MLUObjects);
};

/*!
 * \brief The mlu device context.
 */
class DRAGON_API MLUContext {
 public:
  /*! \brief Constructor */
  MLUContext() : device_id_(0), random_seed_(-1) {}

  /*! \brief Constructor with the device index */
  explicit MLUContext(int device) : device_id_(device) {}

  /*! \brief Constructor with the device option */
  explicit MLUContext(const DeviceOption& option)
      : device_id_(option.device_id()), random_seed_(-1) {
    CHECK_EQ(option.device_type(), PROTO_MLU);
    if (option.has_random_seed()) random_seed_ = int(option.random_seed());
  }

  /*! \brief Allocate a block of device memory */
  static void* New(size_t size) {
    void* data;
    cnrtMalloc(&data, size);
    CHECK(data) << "\nAllocate device memory with " << size << " bytes failed.";
    return data;
  }

  /*! \brief Set a memory block to the given value */
  static void Memset(size_t n, void* ptr, int value = 0) {
    auto stream = objects().default_stream();
    CNRT_CHECK(cnrtMemsetAsync(ptr, value, n, stream));
    SynchronizeStream(stream);
  }

  /*! \brief Set a memory block to the given value asynchronously */
  void MemsetAsync(size_t n, void* ptr, int value = 0) {
    CNRT_CHECK(cnrtMemsetAsync(ptr, value, n, mlu_stream()));
  }

  /*! \brief Copy a memory block to the destination */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t n, void* dest, const void* src) {
    Memcpy<DestContext, SrcContext>(n, dest, src, current_device());
  }

  /*! \brief Copy a memory block to the destination using given device */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t n, void* dest, const void* src, int device) {
    auto stream = objects().default_stream(device);
    CNRT_CHECK(cnrtMemcpyAsync(
        dest, const_cast<void*>(src), n, stream, cnrtMemcpyNoDirection));
    SynchronizeStream(stream);
  }

  /*! \brief Copy a memory block to the destination asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t n, void* dest, const void* src) {
    CNRT_CHECK(cnrtMemcpyAsync(
        dest, const_cast<void*>(src), n, mlu_stream(), cnrtMemcpyNoDirection));
  }

  /*! \brief Synchronize the given stream */
  static void SynchronizeStream(cnrtQueue_t stream) {
    cnrtQueueSync(stream);
    auto err = cnrtGetLastError();
    CHECK_EQ(err, cnrtSuccess) << "\nCNRT Error: " << cnrtGetErrorStr(err);
  }

  /*! \brief Deallocate a device memory block */
  static void Delete(void* ptr) {
    // Synchronize all executing streams.
    CNRT_CHECK(cnrtSyncDevice());
    cnrtFree(ptr);
  }

  /*! \brief Switch to the device and select given stream in current thread */
  void SwitchToDevice(int stream_id = 0) {
    objects().SetDevice(device_id_);
    stream_id_ = stream_id;
  }

  /*! \brief Copy a typed memory block to the destination */
  template <typename T, class DestContext, class SrcContext>
  void Copy(int n, T* dest, const T* src) {
    if (dest == src) return;
    MemcpyAsync<SrcContext, DestContext>(n * sizeof(T), dest, src);
  }

  /*! \brief Wait for the dispatched computation to complete */
  void FinishDeviceComputation() {
    SynchronizeStream(mlu_stream());
  }

  /*! \brief Return the current workspace */
  Workspace* workspace() {
    return objects().workspace(device_id_, stream_id_);
  }

  /*! \brief Return the specified workspace */
  Workspace* workspace(int device, int stream) {
    return objects().workspace(device, stream);
  }

  /*! \brief Return the current mlu stream */
  cnrtQueue_t mlu_stream() {
    return objects().stream(device_id_, stream_id_);
  }

  /*! \brief Return the specified mlu stream */
  cnrtQueue_t mlu_stream(int device, int stream) {
    return objects().stream(device, stream);
  }

  /*! \brief Return the cnrand generator */
  std::pair<cnnlRandGenerator_t, void*> cnrand_generator() {
    return objects().cnrand_generator(device_id_, stream_id_, random_seed());
  }

  /*! \brief Return the cnnl handle */
  cnnlHandle_t cnnl_handle() {
    return objects().cnnl_handle(device_id_, stream_id_);
  }

  /*! \brief Return the device index */
  int device() const {
    return device_id_;
  }

  /*! \brief Return the stream index */
  int stream() const {
    return stream_id_;
  }

  /*! \brief Return the random seed */
  int random_seed() const {
    return random_seed_ >= 0 ? random_seed_
                             : objects().GetRandomSeed(device_id_);
  }

  /*! \brief Return the device index of current thread */
  static int current_device() {
    return objects().GetDevice();
  }

  /*! \brief Return the shared context mutex */
  static std::mutex& mutex();

  /*! \brief Return the thread-local mlu objects */
  static MLUObjects& objects();

  /*! \brief Set the stream index */
  void set_stream(int stream) {
    stream_id_ = stream;
  }

 private:
  /*! \brief The device index */
  int device_id_;

  /*! \brief The stream index */
  int stream_id_ = 0;

  /*! \brief The random seed */
  int random_seed_;
};

#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_CORE_CONTEXT_MLU_H_
