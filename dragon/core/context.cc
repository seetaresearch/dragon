#include "dragon/core/context_cuda.h"
#include "dragon/core/workspace.h"

namespace dragon {

Workspace* CPUContext::workspace() {
  static thread_local Workspace workspace("");
  return &workspace;
}

#ifdef USE_CUDA

CUDAObjects::~CUDAObjects() {
  for (int i = 0; i < CUDA_MAX_DEVICES; i++) {
#ifdef USE_NCCL
    for (auto& comm_iter : nccl_comms_[i]) {
      if (comm_iter.second) {
        NCCL_CHECK(ncclCommDestroy(comm_iter.second));
      }
    }
#endif
#ifdef USE_CUDNN
    for (auto& handle : cudnn_handles_[i]) {
      /*!
       * Temporarily disable the handle destroying,
       * to avoid the segmentation fault in CUDNN v8.
       *
       * if (handle) CUDNN_CHECK(cudnnDestroy(handle));
       */
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
    for (auto& workspace : cuda_workspaces_[i]) {
      if (workspace) delete workspace;
    }
  }
}

Workspace* CUDAObjects::workspace(int device_id, int stream_id) {
  auto& workspaces = cuda_workspaces_[device_id];
  if (workspaces.size() <= (unsigned)stream_id) {
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
  static thread_local CUDAObjects cuda_objects_;
  return cuda_objects_;
}

#endif // USE_CUDA

} // namespace dragon
