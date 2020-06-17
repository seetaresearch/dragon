/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_CONTEXT_CUDA_H_
#define DRAGON_CORE_CONTEXT_CUDA_H_

/* NVIDIA CUDA Environment */

#include "dragon/core/common.h"
#include "dragon/utils/cuda_device.h"
#include "dragon/utils/cudnn_device.h"

namespace dragon {

#ifdef USE_CUDA

class CUDAObject {
 public:
  /*! \brief Default Constructor */
  CUDAObject() {
    for (int i = 0; i < CUDA_MAX_DEVICES; i++) {
      cuda_streams_[i] = vector<cudaStream_t>();
      cublas_handles_[i] = vector<cublasHandle_t>();
#ifdef USE_CUDNN
      cudnn_handles_[i] = vector<cudnnHandle_t>();
#endif
#ifdef USE_NCCL
      nccl_comms_[i] = Map<string, ncclComm_t>();
#endif
    }
  }

  /*! \brief Destructor */
  ~CUDAObject() {
    for (int i = 0; i < CUDA_MAX_DEVICES; i++) {
      for (int j = 0; j < cuda_streams_[i].size(); j++) {
        auto& stream = cuda_streams_[i][j];
        /*!
         * Do not check the stream destroying,
         * error code 29 (driver shutting down) is inevitable.
         */
        if (stream) cudaStreamDestroy(stream);
      }
      for (auto& e : cublas_handles_[i])
        if (e) {
          CUBLAS_CHECK(cublasDestroy_v2(e));
        }
#ifdef USE_CUDNN
      for (auto& e : cudnn_handles_[i])
        if (e) {
          CUDNN_CHECK(cudnnDestroy(e));
        }
#endif
#ifdef USE_NCCL
      for (auto& e : nccl_comms_[i]) {
        /*!
         * Temporarily disable the comm destroying,
         * to avoid an unhandled error.
         */
      }
#endif
    }
  }

  /*! \brief Return the specified cublas handle */
  cublasHandle_t cublas_handle(int device_id, int stream_id) {
    auto& handles = cublas_handles_[device_id];
    if (handles.size() <= (unsigned)stream_id)
      handles.resize(stream_id + 1, nullptr);
    if (!handles[stream_id]) {
      CUDADeviceGuard guard(device_id);
      CUBLAS_CHECK(cublasCreate_v2(&handles[stream_id]));
      CUBLAS_CHECK(
          cublasSetStream_v2(handles[stream_id], stream(device_id, stream_id)));
#if CUDA_VERSION >= 9000
      if (TENSOR_CORE_AVAILABLE()) {
        CUBLAS_CHECK(
            cublasSetMathMode(handles[stream_id], CUBLAS_TENSOR_OP_MATH));
      }
#endif
    }
    return handles[stream_id];
  }

  /*! \brief Return the specified cudnn handle */
#ifdef USE_CUDNN
  cudnnHandle_t cudnn_handle(int device_id, int stream_id) {
    auto& handles = cudnn_handles_[device_id];
    if (handles.size() <= (unsigned)stream_id)
      handles.resize(stream_id + 1, nullptr);
    if (!handles[stream_id]) {
      CUDADeviceGuard guard(device_id);
      CUDNN_CHECK(cudnnCreate(&handles[stream_id]));
      CUDNN_CHECK(
          cudnnSetStream(handles[stream_id], stream(device_id, stream_id)));
    }
    return handles[stream_id];
  }
#endif

  /*! \brief Return the specified nccl comm */
#ifdef USE_NCCL
  ncclComm_t nccl_comm(
      int device_id,
      const string& cache_key,
      ncclUniqueId* comm_uuid,
      int comm_size,
      int comm_rank) {
    auto& comms = nccl_comms_[device_id];
    auto find_iter = comms.find(cache_key);
    if (find_iter != comms.end()) return find_iter->second;
    if (comm_uuid == nullptr) return nullptr;
    CUDADeviceGuard guard(device_id);
    NCCL_CHECK(
        ncclCommInitRank(&comms[cache_key], comm_size, *comm_uuid, comm_rank));
    return comms[cache_key];
  }
#endif

  /*! \brief Return the default cuda stream of current device */
  cudaStream_t default_stream() {
    return stream(CUDA_GET_DEVICE(), 0);
  }

  /*! \brief Return the default cuda stream of given device */
  cudaStream_t default_stream(int device_id) {
    return stream(device_id, 0);
  }

  /*! \brief Return the specified cuda stream */
  cudaStream_t stream(int device_id, int stream_id) {
    auto& streams = cuda_streams_[device_id];
    if (streams.size() <= (unsigned)stream_id)
      streams.resize(stream_id + 1, nullptr);
    if (!streams[stream_id]) {
      CUDADeviceGuard guard(device_id);
      unsigned int flags =
          !stream_id ? cudaStreamDefault : cudaStreamNonBlocking;
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[stream_id], flags));
    }
    return streams[stream_id];
  }

  vector<cudaStream_t> cuda_streams_[CUDA_MAX_DEVICES];
  vector<cublasHandle_t> cublas_handles_[CUDA_MAX_DEVICES];
#ifdef USE_CUDNN
  vector<cudnnHandle_t> cudnn_handles_[CUDA_MAX_DEVICES];
#endif

#ifdef USE_NCCL
  Map<string, ncclComm_t> nccl_comms_[CUDA_MAX_DEVICES];
#endif

  bool cudnn_enabled_ = true;
  bool cudnn_benchmark_ = false;
};

class DRAGON_API CUDAContext {
 public:
  /*! \brief Default Constructor */
  explicit CUDAContext(const DeviceOption& option)
      : device_id_(option.device_id()),
        random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : DEFAULT_RNG_SEED) {
    CHECK_EQ(option.device_type(), PROTO_CUDA);
  }

  /*! \brief Constructor with the specified device index */
  explicit CUDAContext(int device_id = 0)
      : device_id_(device_id), random_seed_(DEFAULT_RNG_SEED) {}

  /*! \brief Alloc the memory */
  static void* New(size_t nbytes) {
    void* data;
    cudaMalloc(&data, nbytes);
    CHECK(data) << "\nAllocate cuda memory with " << nbytes << " bytes failed.";
    return data;
  }

  /*! \brief Zero-Reset the memory */
  static void Memset(size_t nbytes, void* ptr) {
    auto stream = object()->default_stream();
    CUDA_CHECK(cudaMemsetAsync(ptr, 0, nbytes, stream));
    SyncStream(stream);
  }

  /*! \brief Copy the memory */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t nbytes, void* dest, const void* src) {
    Memcpy<DestContext, SrcContext>(nbytes, dest, src, current_device());
  }

  /*! \brief Copy the memory using specific stream */
  template <class DestContext, class SrcContext>
  static void
  Memcpy(size_t nbytes, void* dest, const void* src, int device_id) {
    auto stream = object()->default_stream(device_id);
    CUDA_CHECK(cudaMemcpyAsync(dest, src, nbytes, cudaMemcpyDefault, stream));
    SyncStream(stream);
  }

  /*! \brief Synchronize the specified cuda stream */
  static void SyncStream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    auto error = cudaGetLastError();
    CHECK_EQ(error, cudaSuccess)
        << "\nCUDA Error: " << cudaGetErrorString(error);
  }

  /*! \brief Free the memory */
  static void Delete(void* data) {
    cudaFree(data);
  }

  /*! \brief Zero-Reset the memory asynchronously */
  void MemsetAsync(size_t nbytes, void* ptr) {
    CUDA_CHECK(cudaMemsetAsync(ptr, 0, nbytes, cuda_stream()));
  }

  /*! \brief Copy the memory asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t nbytes, void* dest, const void* src) {
    CUDA_CHECK(
        cudaMemcpyAsync(dest, src, nbytes, cudaMemcpyDefault, cuda_stream()));
  }

  /*! \brief Switch to the device with the given stream */
  void SwitchToDevice(const int stream_id) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    stream_id_ = stream_id;
  }

  /*! \brief Switch to the device of this context */
  void SwitchToDevice() {
    SwitchToDevice(0);
  }

  /*! \brief Copy the memory with given type asynchronously */
  template <typename T, class DestContext, class SrcContext>
  void Copy(int n, T* dest, const T* src) {
    if (dest == src) return;
    MemcpyAsync<SrcContext, DestContext>(n * sizeof(T), dest, src);
  }

  /*! \brief Synchronize the dispatched operations */
  void FinishDeviceComputation() {
    SyncStream(cuda_stream());
  }

  /*! \brief Return the internal cuda stream */
  cudaStream_t cuda_stream() {
    return cuda_stream(device_id_, stream_id_);
  }

  /*! \brief Return the specified cuda stream */
  cudaStream_t cuda_stream(int device_id, int stream_id) {
    return object()->stream(device_id, stream_id);
  }

  /*! \brief Return the internal cublas handle */
  cublasHandle_t cublas_handle() {
    return object()->cublas_handle(device_id_, stream_id_);
  }

  /*! \brief Return the internal cuda random generator */
  curandGenerator_t& curand_generator() {
    if (!curand_generator_) {
      CUDADeviceGuard guard(device_id_);
      CURAND_CHECK(
          curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
      CURAND_CHECK(
          curandSetPseudoRandomGeneratorSeed(curand_generator_, random_seed_));
    }
    CURAND_CHECK(curandSetStream(curand_generator_, cuda_stream()));
    return curand_generator_;
  }

  /*! \brief Return the internal cudnn handle */
#ifdef USE_CUDNN
  cudnnHandle_t cudnn_handle() {
    return object()->cudnn_handle(device_id_, stream_id_);
  }
#endif

  /*! \brief Return the device index of this context */
  int device_id() const {
    return device_id_;
  }

  /*! \brief Return the device index of current thread */
  static int current_device() {
    return CUDA_GET_DEVICE();
  }

  /*! \brief Return the stream id */
  int stream_id() const {
    return stream_id_;
  }

  /*! \brief Return the global context locker */
  static std::mutex& mutex();

  /*! \brief Return the thread local cuda object */
  static CUDAObject* object();

  /*! \brief Return the internal random generator */
  std::mt19937* rand_generator() {
    if (!rand_generator_.get()) {
      rand_generator_.reset(new std::mt19937(random_seed_));
    }
    return rand_generator_.get();
  }

  /*! \brief Set the stream id */
  void set_stream_id(int stream_id) {
    stream_id_ = stream_id;
  }

 private:
  int device_id_, stream_id_ = 0, random_seed_;
  unique_ptr<std::mt19937> rand_generator_;
  curandGenerator_t curand_generator_ = nullptr;
};

#else // USE_CUDA

class DRAGON_API CUDAContext {
 public:
  /*! \brief Default Constructor */
  explicit CUDAContext(const DeviceOption& option) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Constructor with the specified device id */
  explicit CUDAContext(const int device_id = 0) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Alloc the memory */
  static void* New(size_t nbytes) {
    CUDA_NOT_COMPILED;
    return nullptr;
  }

  /*! \brief Zero-Reset the memory */
  static void Memset(size_t nbytes, void* ptr) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Copy the memory */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t nbytes, void* dest, const void* src) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Copy the memory using specific stream */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t nbytes, void* dst, const void* src, int device_id) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Free the memory */
  static void Delete(void* data) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Zero-Reset the memory asynchronously */
  void MemsetAsync(size_t nbytes, void* ptr) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Copy the memory asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t nbytes, void* dest, const void* src) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Switch to the device with the given stream */
  void SwitchToDevice(int stream_id) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Switch to the device of this context */
  void SwitchToDevice() {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Synchronize the dispatched operations */
  void FinishDeviceComputation() {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Return the device index of this context */
  int device_id() const {
    return 0;
  }

  /*! \brief Return the device index of current thread */
  static int current_device() {
    return 0;
  }

  /*! \brief Return the stream id */
  int stream_id() const {
    return 0;
  }

  /*! \brief Set the stream id */
  void set_stream_id(int stream_id) {}
};

#endif // USE_CUDA

} // namespace dragon

#endif // DRAGON_CORE_CONTEXT_CUDA_H_
