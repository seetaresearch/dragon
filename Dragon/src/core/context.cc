#include "core/context.h"
#include "core/context_cuda.h"

namespace dragon {

//  cpu <- gpu
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

//  gpu <- cpu
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

}    // namespace dragon