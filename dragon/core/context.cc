#include "context_cuda.h"

namespace dragon {

#ifdef USE_CUDA

std::mutex& CUDAContext::mutex() {
  static std::mutex m;
  return m;
}

CUDAObject* CUDAContext::object() {
  static TLS_OBJECT CUDAObject* cuda_object_;
  if (!cuda_object_) cuda_object_ = new CUDAObject();
  return cuda_object_;
}

#endif // USE_CUDA

} // namespace dragon
