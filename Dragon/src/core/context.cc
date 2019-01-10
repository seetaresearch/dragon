#include "core/context.h"
#include "core/context_cuda.h"

namespace dragon {

// CPU <- CUDA
template<> void CPUContext::Memcpy<CPUContext, CUDAContext>(
    size_t                  nbytes,
    void*                   dst,
    const void*             src) {
#ifdef WITH_CUDA
     CUDAContext::Memcpy<CPUContext, CUDAContext>(nbytes, dst, src);
#else
    LOG(FATAL) << "CUDA was not compiled.";
#endif
}

// CUDA <- CPU
template<> void CPUContext::Memcpy<CUDAContext, CPUContext>(
    size_t                  nbytes,
    void*                   dst,
    const void*             src) {
#ifdef WITH_CUDA
    CUDAContext::Memcpy<CUDAContext, CPUContext>(nbytes, dst, src);
#else
    LOG(FATAL) << "CUDA was not compiled.";
#endif
}

}  // namespace dragon