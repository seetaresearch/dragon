#include "core/tensor.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _MaskedSelectGrad(
    const int               num_indices,
    const int64_t*          indices,
    const T*                dy,
    T*                      dx) {
    for (int i = 0; i < num_indices; ++i) {
        dx[indices[i]] = dy[i];
    }
}

/* Kernel Launchers */

#define DEFINE_MASKED_KERNEL_LAUNCHER(T) \
    template <> void MaskedSelect<T, CPUContext>( \
        const int               count, \
        const uint8_t*          mask, \
        const T*                x, \
        Tensor*                 indices, \
        Tensor*                 scratch, \
        Tensor*                 y, \
        CPUContext*             ctx) { \
        int64_t nelements = 0; \
        int64_t n, last = -1, y_ofs = 0; \
        for (int i = 0; i < count; ++i) \
            if (mask[i]) ++nelements; \
        auto* value = y == nullptr ? nullptr : y \
            ->Reshape({ nelements }) \
            ->mutable_data<T, CPUContext>(); \
        auto* index = indices \
            ->Reshape({ nelements }) \
            ->mutable_data<int64_t, CPUContext>(); \
        for (int64_t i = 0;; ++i) { \
            if (last != -1 && ((i >= count) || !mask[i])) { \
                n = i - last; \
                if (value != nullptr) { \
                    auto* src = x + last; \
                    auto* dst = value + y_ofs; \
                    math::Copy(n, src, dst, ctx); \
                } \
                y_ofs += n; last = -1; \
            } \
            if (i >= count) break; \
            if (mask[i]) { \
                *(index++) = i; \
                if (last == -1) last = i; \
            } \
        } \
    }

#define DEFINE_MASKED_GRAD_KERNEL_LAUNCHER(T) \
    template <> void MaskedSelectGrad<T, CPUContext>( \
        const int               count, \
        const int               num_indices, \
        const int64_t*          indices, \
        const T*                dy, \
        T*                      dx, \
        CPUContext*             ctx) { \
        math::Set(count, cast::to<T>(0.f), dx, ctx); \
        _MaskedSelectGrad(num_indices, indices, dy, dx); \
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