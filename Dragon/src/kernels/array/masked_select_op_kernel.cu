#ifdef WITH_CUDA

#include "core/tensor.h"
#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/cub_device.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _MaskedSelectByIndex(
    const int               nthreads,
    const int64_t*          indices,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = x[indices[i]];
    }
}

template <typename T>
__global__ void _MaskedSelectGrad(
    const int               nthreads,
    const int64_t*          indices,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        dx[indices[i]] = dy[i];
    }
}

/* Kernel Launchers */

#define DEFINE_MASKED_KERNEL_LAUNCHER(T) \
    template <> void MaskedSelect<T, CUDAContext>( \
        const int               count, \
        const uint8_t*          mask, \
        const T*                x, \
        Tensor*                 indices, \
        Tensor*                 scratch, \
        Tensor*                 y, \
        CUDAContext*            ctx) { \
        auto* i = indices \
            ->Reshape({ count + 1 }) \
            ->mutable_data<int64_t, CUDAContext>(); \
        auto* n = (int*)(i + count); \
        size_t nbytes = 0; int nelements; \
        cub::CountingInputIterator<int> itr(0); \
        cub::DeviceSelect::Flagged( \
            nullptr, nbytes, \
            itr, mask, i, n, count, \
            ctx->cuda_stream() \
        ); \
        auto* storage = scratch \
            ->Reshape({ (int64_t)nbytes }) \
            ->mutable_data<uint8_t, CUDAContext>(); \
        cub::DeviceSelect::Flagged( \
            storage, nbytes, \
            itr, mask, i, n, count, \
            ctx->cuda_stream() \
        ); \
        ctx->FinishDeviceCompution(); \
        ctx->Memcpy<CPUContext, CUDAContext>( \
            sizeof(int), &nelements, n); \
        indices->Reshape({ nelements }); \
        if (y == nullptr) return; \
        auto* value = y \
            ->Reshape({ nelements }) \
            ->mutable_data<T, CUDAContext>(); \
        _MaskedSelectByIndex \
            <<< CUDA_BLOCKS(nelements), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            nelements, i, x, value \
        ); \
    }

#define DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(T) \
    template <> void MaskedSelectGrad<T, CUDAContext>( \
        const int               count, \
        const int               num_indices, \
        const int64_t*          indices, \
        const T*                dy, \
        T*                      dx, \
        CUDAContext*            ctx) { \
        math::Set(count, cast::to<T>(0.f), dx, ctx); \
        _MaskedSelectGrad \
            <<< CUDA_BLOCKS(num_indices), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            num_indices, indices, dy, dx \
        ); \
    }

DEFINE_MASKED_KERNEL_LAUNCHER(bool);
DEFINE_MASKED_KERNEL_LAUNCHER(int8_t);
DEFINE_MASKED_KERNEL_LAUNCHER(uint8_t);
DEFINE_MASKED_KERNEL_LAUNCHER(int);
DEFINE_MASKED_KERNEL_LAUNCHER(int64_t);
DEFINE_MASKED_KERNEL_LAUNCHER(float16);
DEFINE_MASKED_KERNEL_LAUNCHER(float);
DEFINE_MASKED_KERNEL_LAUNCHER(double);

DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(bool);
DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(int);
DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(float);
DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_MASKED_KERNEL_LAUNCHER
#undef DEFINE_MASKED_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA