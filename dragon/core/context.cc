#include "context_cuda.h"

namespace dragon {

#ifdef USE_CUDA

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
