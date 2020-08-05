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

#ifndef DRAGON_CORE_CONTEXT_CUDA_H_
#define DRAGON_CORE_CONTEXT_CUDA_H_

#include "dragon/core/common.h"
#include "dragon/utils/cuda_device.h"
#include "dragon/utils/cudnn_device.h"

namespace dragon {

#ifdef USE_CUDA

class CUDAObjects {
 public:
  /*! \brief Default Constructor */
  CUDAObjects() {
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
  ~CUDAObjects() {
    for (int i = 0; i < CUDA_MAX_DEVICES; i++) {
#ifdef USE_NCCL
      for (auto& comm : nccl_comms_[i]) {
        /*!
         * Temporarily disable the comm destroying,
         * to avoid an unhandled error.
         */
      }
#endif
#ifdef USE_CUDNN
      for (auto& handle : cudnn_handles_[i]) {
        if (handle) CUDNN_CHECK(cudnnDestroy(handle));
      }
#endif
      for (auto& handle : cublas_handles_[i]) {
        if (handle) CUBLAS_CHECK(cublasDestroy(handle));
      }
      for (int j = 0; j < cuda_streams_[i].size(); j++) {
        auto& stream = cuda_streams_[i][j];
        /*!
         * Do not check the stream destroying,
         * error code 29 (driver shutting down) is inevitable.
         */
        if (stream) cudaStreamDestroy(stream);
      }
    }
  }

  /*! \brief Return the specified cublas handle */
  cublasHandle_t cublas_handle(int device_id, int stream_id) {
    auto& handles = cublas_handles_[device_id];
    if (handles.size() <= (unsigned)stream_id) {
      handles.resize(stream_id + 1, nullptr);
    }
    if (!handles[stream_id]) {
      CUDADeviceGuard guard(device_id);
      CUBLAS_CHECK(cublasCreate(&handles[stream_id]));
      auto& handle = handles[stream_id];
      CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
      CUBLAS_CHECK(cublasSetStream(handle, stream(device_id, stream_id)));
#if CUDA_VERSION >= 9000
      if (TENSOR_CORE_AVAILABLE()) {
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
      }
#endif
    }
    return handles[stream_id];
  }

  /*! \brief Return the specified cudnn handle */
#ifdef USE_CUDNN
  cudnnHandle_t cudnn_handle(int device_id, int stream_id) {
    auto& handles = cudnn_handles_[device_id];
    if (handles.size() <= (unsigned)stream_id) {
      handles.resize(stream_id + 1, nullptr);
    }
    if (!handles[stream_id]) {
      CUDADeviceGuard guard(device_id);
      CUDNN_CHECK(cudnnCreate(&handles[stream_id]));
      auto& handle = handles[stream_id];
      CUDNN_CHECK(cudnnSetStream(handle, stream(device_id, stream_id)));
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
          stream_id == 0 ? cudaStreamDefault : cudaStreamNonBlocking;
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

 private:
  DISABLE_COPY_AND_ASSIGN(CUDAObjects);
};

/*!
 * \brief The cuda device context.
 */
class DRAGON_API CUDAContext {
 public:
  /*! \brief Default constructor */
  CUDAContext() : device_id_(0), random_seed_(DEFAULT_RNG_SEED) {}

  /*! \brief Constructor with the device index */
  explicit CUDAContext(int device) : device_id_(device) {}

  /*! \brief Constructor with the device option */
  explicit CUDAContext(const DeviceOption& option)
      : device_id_(option.device_id()),
        random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : DEFAULT_RNG_SEED) {
    CHECK_EQ(option.device_type(), PROTO_CUDA);
  }

  /*! \brief Allocate a block of memory */
  static void* New(size_t size) {
    void* data;
    cudaMalloc(&data, size);
    CHECK(data) << "\nAllocate cuda memory with " << size << " bytes failed.";
    return data;
  }

  /*! \brief Set a memory block to the given value */
  static void Memset(size_t n, void* ptr, int value = 0) {
    auto stream = objects().default_stream();
    CUDA_CHECK(cudaMemsetAsync(ptr, value, n, stream));
    SynchronizeStream(stream);
  }

  /*! \brief Set a memory block to the given value asynchronously */
  void MemsetAsync(size_t n, void* ptr, int value = 0) {
    CUDA_CHECK(cudaMemsetAsync(ptr, value, n, cuda_stream()));
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
    CUDA_CHECK(cudaMemcpyAsync(dest, src, n, cudaMemcpyDefault, stream));
    SynchronizeStream(stream);
  }

  /*! \brief Copy a memory block to the destination asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t n, void* dest, const void* src) {
    CUDA_CHECK(cudaMemcpyAsync(dest, src, n, cudaMemcpyDefault, cuda_stream()));
  }

  /*! \brief Synchronize the given stream */
  static void SynchronizeStream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    auto err = cudaGetLastError();
    CHECK_EQ(err, cudaSuccess) << "\nCUDA Error: " << cudaGetErrorString(err);
  }

  /*! \brief Deallocate a memory block */
  static void Delete(void* ptr) {
    cudaFree(ptr);
  }

  /*! \brief Switch to the device in current thread */
  void SwitchToDevice() {
    SwitchToDevice(0);
  }

  /*! \brief Switch to the device and select given stream in current thread */
  void SwitchToDevice(int stream) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    stream_id_ = stream;
  }

  /*! \brief Copy a typed memory block to the destination */
  template <typename T, class DestContext, class SrcContext>
  void Copy(int n, T* dest, const T* src) {
    if (dest == src) return;
    MemcpyAsync<SrcContext, DestContext>(n * sizeof(T), dest, src);
  }

  /*! \brief Wait for the dispatched computation to complete */
  void FinishDeviceComputation() {
    SynchronizeStream(cuda_stream());
  }

  /*! \brief Return the cuda stream */
  cudaStream_t cuda_stream() {
    return cuda_stream(device_id_, stream_id_);
  }

  /*! \brief Return the specified cuda stream */
  cudaStream_t cuda_stream(int device, int stream) {
    return objects().stream(device, stream);
  }

  /*! \brief Return the cublas handle */
  cublasHandle_t cublas_handle() {
    return objects().cublas_handle(device_id_, stream_id_);
  }

  /*! \brief Return the curand generator */
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

  /*! \brief Return the cudnn handle */
#ifdef USE_CUDNN
  cudnnHandle_t cudnn_handle() {
    return objects().cudnn_handle(device_id_, stream_id_);
  }
#endif

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
    return CUDA_GET_DEVICE();
  }

  /*! \brief Return the shared context mutex */
  static std::mutex& mutex();

  /*! \brief Return the thread-local cuda objects */
  static CUDAObjects& objects();

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
  curandGenerator_t curand_generator_ = nullptr;
};

#else // USE_CUDA

class DRAGON_API CUDAContext {
 public:
  /*! \brief Default constructor */
  explicit CUDAContext() {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Constructor with the device index */
  explicit CUDAContext(int device) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Constructor with the device option */
  explicit CUDAContext(const DeviceOption& option) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Allocate a block of memory */
  static void* New(size_t nbytes) {
    CUDA_NOT_COMPILED;
    return nullptr;
  }

  /*! \brief Set a memory block to the given value */
  static void Memset(size_t nbytes, void* ptr, int value = 0) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Set a memory block to the given value asynchronously */
  void MemsetAsync(size_t nbytes, void* ptr, int value = 0) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Copy a memory block to the destination */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t nbytes, void* dest, const void* src) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Copy a memory block to the destination using given device */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t nbytes, void* dst, const void* src, int device_id) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Deallocate a memory block */
  static void Delete(void* ptr) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Copy the memory asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t nbytes, void* dest, const void* src) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Switch to the device in current thread */
  void SwitchToDevice() {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Switch to the device and select given stream in current thread */
  void SwitchToDevice(int stream_id) {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Wait for the dispatched computation to complete */
  void FinishDeviceComputation() {
    CUDA_NOT_COMPILED;
  }

  /*! \brief Return the device index */
  int device() const {
    return 0;
  }

  /*! \brief Return the device index of current thread */
  static int current_device() {
    return 0;
  }

  /*! \brief Return the stream index */
  int stream() const {
    return 0;
  }
};

#endif // USE_CUDA

} // namespace dragon

#endif // DRAGON_CORE_CONTEXT_CUDA_H_
