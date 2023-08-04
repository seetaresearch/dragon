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
#include "dragon/utils/device/common_cuda.h"
#include "dragon/utils/device/common_cudnn.h"
#include "dragon/utils/device/common_nccl.h"

namespace dragon {

#ifdef USE_CUDA

class Workspace;

class DRAGON_API CUDAObjects {
 public:
  /*! \brief Constructor */
  CUDAObjects() {
    for (int i = 0; i < CUDA_MAX_DEVICES; ++i) {
      random_seeds_[i] = DEFAULT_RNG_SEED;
      streams_[i] = vector<cudaStream_t>();
      workspaces_[i] = vector<Workspace*>();
      cublas_handles_[i] = vector<cublasHandle_t>();
      curand_generators_[i] = Map<string, curandGenerator_t>();
#ifdef USE_CUDNN
      cudnn_handles_[i] = vector<cudnnHandle_t>();
#endif
#ifdef USE_NCCL
      nccl_comms_[i] = Map<string, ncclComm_t>();
#endif
    }
  }

  /*! \brief Destructor */
  ~CUDAObjects();

  /*! \brief Set the current device */
  void SetDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
  }

  /*! \brief Set the random seed for given device */
  void SetRandomSeed(int device_id, int seed) {
    if (device_id < CUDA_MAX_DEVICES) random_seeds_[device_id] = seed;
  }

  /*! \brief Return the current cuda device */
  int GetDevice() {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    return device_id;
  }

  /*! \brief Return the random seed of given device */
  int GetRandomSeed(int device_id) const {
    return (device_id < CUDA_MAX_DEVICES) ? random_seeds_[device_id]
                                          : DEFAULT_RNG_SEED;
  }

  /*! \brief Return the specified cublas handle */
  cublasHandle_t cublas_handle(int device_id, int stream_id);

  /*! \brief Return the specified curand generator */
  curandGenerator_t curand_generator(int device_id, int stream_id, int seed);

#ifdef USE_CUDNN
  /*! \brief Return the specified cudnn handle */
  cudnnHandle_t cudnn_handle(int device_id, int stream_id);
#endif

#ifdef USE_NCCL
  /*! \brief Return the specified nccl comm */
  ncclComm_t nccl_comm(
      int device_id,
      const string& cache_key,
      ncclUniqueId* comm_uuid,
      int comm_size,
      int comm_rank);
#endif

  /*! \brief Return the default cuda stream of current device */
  cudaStream_t default_stream() {
    return stream(GetDevice(), 0);
  }

  /*! \brief Return the default cuda stream of given device */
  cudaStream_t default_stream(int device_id) {
    return stream(device_id, 0);
  }

  /*! \brief Return the specified cuda stream */
  cudaStream_t stream(int device_id, int stream_id) {
    auto& streams = streams_[device_id];
    if (streams.size() <= unsigned(stream_id)) {
      streams.resize(stream_id + 1, nullptr);
    }
    if (!streams[stream_id]) {
      CUDADeviceGuard guard(device_id);
      auto flags = stream_id == 0 ? cudaStreamDefault : cudaStreamNonBlocking;
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[stream_id], flags));
    }
    return streams[stream_id];
  }

  /*! \brief Return the workspace of specified cuda stream */
  Workspace* workspace(int device_id, int stream_id);

  /*! \brief The random seed for all devices */
  int random_seeds_[CUDA_MAX_DEVICES];

  /*! \brief The created streams for all devices */
  vector<cudaStream_t> streams_[CUDA_MAX_DEVICES];

  /*! \brief The created workspaces for all devices */
  vector<Workspace*> workspaces_[CUDA_MAX_DEVICES];

  /*! \brief The created cublas handles for all devices */
  vector<cublasHandle_t> cublas_handles_[CUDA_MAX_DEVICES];

  /*! \brief The created curand generators for all devices */
  Map<string, curandGenerator_t> curand_generators_[CUDA_MAX_DEVICES];

#ifdef USE_CUDNN
  /*! \brief The created cudnn handles for all devices */
  vector<cudnnHandle_t> cudnn_handles_[CUDA_MAX_DEVICES];
#endif

#ifdef USE_NCCL
  /*! \brief The created nccl comms for all devices */
  Map<string, ncclComm_t> nccl_comms_[CUDA_MAX_DEVICES];
#endif

  /*! \brief The flag that allows cuBLAS TF32 math type or not */
  bool cublas_allow_tf32_ = false;

  /*! \brief The flag that uses cuDNN or not */
  bool cudnn_enabled_ = true;

  /*! \brief The flag that benchmarks fastest cuDNN algorithms or not */
  bool cudnn_benchmark_ = false;

  /*! \brief The flag that selects deterministic cuDNN algorithms or not */
  bool cudnn_deterministic_ = false;

  /*! \brief The flag that allows cuDNN TF32 math type or not */
  bool cudnn_allow_tf32_ = false;

 private:
  DISABLE_COPY_AND_ASSIGN(CUDAObjects);
};

/*!
 * \brief The cuda device context.
 */
class DRAGON_API CUDAContext {
 public:
  /*! \brief Constructor */
  CUDAContext() : device_id_(0), random_seed_(-1) {}

  /*! \brief Constructor with the device index */
  explicit CUDAContext(int device) : device_id_(device) {}

  /*! \brief Constructor with the device option */
  explicit CUDAContext(const DeviceOption& option)
      : device_id_(option.device_id()), random_seed_(-1) {
    CHECK_EQ(option.device_type(), PROTO_CUDA);
    if (option.has_random_seed()) random_seed_ = int(option.random_seed());
  }

  /*! \brief Allocate a block of device memory */
  static void* New(size_t size) {
    void* data;
    cudaMalloc(&data, size);
    CHECK(data) << "\nAllocate device memory with " << size << " bytes failed.";
    return data;
  }

  /*! \brief Allocate a block of host memory */
  static void* NewHost(size_t size) {
    void* data;
    cudaMallocHost(&data, size);
    CHECK(data) << "\nAllocate host memory with " << size << " bytes failed.";
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

  /*! \brief Deallocate a device memory block */
  static void Delete(void* ptr) {
    cudaFree(ptr);
  }

  /*! \brief Deallocate a host memory block */
  static void DeleteHost(void* ptr) {
    cudaFreeHost(ptr);
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
    SynchronizeStream(cuda_stream());
  }

  /*! \brief Return the current workspace */
  Workspace* workspace() {
    return objects().workspace(device_id_, stream_id_);
  }

  /*! \brief Return the specified workspace */
  Workspace* workspace(int device, int stream) {
    return objects().workspace(device, stream);
  }

  /*! \brief Return the current cuda stream */
  cudaStream_t cuda_stream() {
    return objects().stream(device_id_, stream_id_);
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
  curandGenerator_t curand_generator() {
    return objects().curand_generator(device_id_, stream_id_, random_seed());
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

  /*! \brief Return the thread-local cuda objects */
  static CUDAObjects& objects();

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

#endif // USE_CUDA

} // namespace dragon

#endif // DRAGON_CORE_CONTEXT_CUDA_H_
