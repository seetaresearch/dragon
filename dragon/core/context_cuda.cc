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
      // Temporarily disable destroying to avoid segmentation fault.
      // if (handle) CUDNN_CHECK(cudnnDestroy(handle));
    }
#endif
    for (auto& handle : cublas_handles_[device_id]) {
      if (handle) CUBLAS_CHECK(cublasDestroy(handle));
    }
    for (auto& stream : streams_[device_id]) {
      // Do not check the stream destroying.
      if (stream) cudaStreamDestroy(stream);
    }
    for (auto& workspace : workspaces_[device_id]) {
      if (workspace) delete workspace;
    }
  }
}

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
