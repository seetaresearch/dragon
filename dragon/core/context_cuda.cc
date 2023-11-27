#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/core/workspace.h"

namespace dragon {

CUDAObjects::~CUDAObjects() {
  for (int device_id = 0; device_id < CUDA_MAX_DEVICES; ++device_id) {
#ifdef USE_NCCL
    for (auto& iter : nccl_comms_[device_id]) {
      if (iter.second) NCCL_CHECK(ncclCommDestroy(iter.second));
    }
#endif
#ifdef USE_CUDNN
    for (auto& handle : cudnn_handles_[device_id]) {
      if (handle) CUDNN_CHECK(cudnnDestroy(handle));
    }
#endif
    for (auto& handle : cublas_handles_[device_id]) {
      if (handle) CUBLAS_CHECK(cublasDestroy(handle));
    }
    for (auto& iter : curand_generators_[device_id]) {
      if (iter.second) CURAND_CHECK(curandDestroyGenerator(iter.second));
    }
    for (auto& stream : streams_[device_id]) {
      if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    }
    for (auto& workspace : workspaces_[device_id]) {
      if (workspace) delete workspace;
    }
  }
}

cudaStream_t CUDAObjects::stream(int device_id, int stream_id) {
  auto& streams = streams_[device_id];
  if (streams.size() <= unsigned(stream_id)) {
    streams.resize(stream_id + 1, nullptr);
  }
  if (!streams[stream_id]) {
    CUDADeviceGuard guard(device_id);
    const unsigned int flags = cudaStreamNonBlocking;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[stream_id], flags));
  }
  return streams[stream_id];
}

cublasHandle_t CUDAObjects::cublas_handle(int device_id, int stream_id) {
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
  }
  auto& handle = handles[stream_id];
#if CUDA_VERSION >= 11000
  if (cublas_allow_tf32_) {
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  } else {
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
#endif
  return handle;
}

curandGenerator_t
CUDAObjects::curand_generator(int device_id, int stream_id, int seed) {
  auto& generators = curand_generators_[device_id];
  const string key = str::to(stream_id) + "/RNGState:" + str::to(seed);
  auto find_iter = generators.find(key);
  if (find_iter != generators.end()) return find_iter->second;
  CUDADeviceGuard guard(device_id);
  const auto rng_type = CURAND_RNG_PSEUDO_DEFAULT;
  CURAND_CHECK(curandCreateGenerator(&generators[key], rng_type));
  auto& generator = generators[key];
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));
  CURAND_CHECK(curandSetStream(generator, stream(device_id, stream_id)));
  return generator;
}

#ifdef USE_CUDNN
cudnnHandle_t CUDAObjects::cudnn_handle(int device_id, int stream_id) {
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

#ifdef USE_NCCL
ncclComm_t CUDAObjects::nccl_comm(
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

Workspace* CUDAObjects::workspace(int device_id, int stream_id) {
  auto& workspaces = workspaces_[device_id];
  if (workspaces.size() <= unsigned(stream_id)) {
    workspaces.resize(stream_id + 1, nullptr);
  }
  if (!workspaces[stream_id]) {
    workspaces[stream_id] = new Workspace("");
  }
  return workspaces[stream_id];
}

std::mutex& CUDAContext::mutex() {
  static std::mutex m;
  return m;
}

CUDAObjects& CUDAContext::objects() {
  static thread_local CUDAObjects objects_;
  return objects_;
}

} // namespace dragon

#endif // USE_CUDA
