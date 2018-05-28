#include "core/context.h"
#include "core/context_cuda.h"

namespace dragon {

CPUObject CPUContext::cpu_object_;
#ifdef WITH_CUDA
CUDAObject CUDAContext::cuda_object_;
#endif // WITH_CUDA

//  cpu <- gpu
template<> void CPUContext::Memcpy<CPUContext, CUDAContext>(
    size_t nbytes, void* dst, const void* src) {
#ifdef WITH_CUDA
    CUDAContext ctx(CUDA_POINTER_DEVICE(src));
    ctx.Memcpy<CPUContext, CUDAContext>(nbytes, dst, src);
#else
    LOG(FATAL) << "CUDA was not compiled.";
#endif
}

//  gpu <- cpu
template<> void CPUContext::Memcpy<CUDAContext, CPUContext>(
    size_t nbytes, void* dst, const void* src) {
#ifdef WITH_CUDA
        CUDAContext ctx(CUDA_POINTER_DEVICE(dst));
        ctx.Memcpy<CUDAContext, CPUContext>(nbytes, dst, src);
#else
    LOG(FATAL) << "CUDA was not compiled.";
#endif
}

}    // namespace dragon