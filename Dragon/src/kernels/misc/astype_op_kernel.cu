#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Astype <Ta = ?, Tb = ?, Device = CUDA> */

template <typename Ta, typename Tb>
__global__ void _TypeA2B(
    const int               count,
    const Ta*               a,
    Tb*                     b) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        b[idx] = a[idx];
    }
}

#define DEFINE_TYPE_A2B(type_a, type_b) \
    template <> void TypeA2B<type_a, type_b, CUDAContext>( \
        const int           count, \
        const               type_a* a, \
        type_b*             b, \
        CUDAContext*        ctx) { \
        _TypeA2B<type_a, type_b> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, a, b); \
    }

#define DEFINE_TYPE_A2ALL(type_a) \
    DEFINE_TYPE_A2B(type_a, float); \
    DEFINE_TYPE_A2B(type_a, double); \
    DEFINE_TYPE_A2B(type_a, int); \
    DEFINE_TYPE_A2B(type_a, int64_t); \
    DEFINE_TYPE_A2B(type_a, uint8_t)

DEFINE_TYPE_A2ALL(float);
DEFINE_TYPE_A2ALL(double);
DEFINE_TYPE_A2ALL(int);
DEFINE_TYPE_A2ALL(int64_t);
DEFINE_TYPE_A2ALL(uint8_t);

/*! Astype <Ta = float16, Tb = ?, Device = CUDA> */

__global__ void _TypeHalf2Float(
    const int               count,
    const half*             a,
    float*                  b) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        b[idx] = __half2float(a[idx]);
    }
}

__global__ void _TypeFloat2Half(
    const int               count,
    const float*            a,
    half*                   b) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        b[idx] = __float2half(a[idx]);
    }
}

__global__ void _TypeHalf2Half(
    const int               count,
    const half*             a,
    half*                   b) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        b[idx] = a[idx];
    }
}

#define DEFINE_TYPE_DISABLE_FP16(type) \
    template <> void TypeA2B<float16, type, CUDAContext>( \
        const int           count, \
        const float16*      a, \
        type*               b, \
        CUDAContext*        ctx) { \
        LOG(FATAL) << "CUDAContext has not implemented: float16 -> " \
                   << TypeMetaToString(TypeMeta::Make<type>()); \
    } \
    template <> void TypeA2B<type, float16, CUDAContext>( \
        const int           count, \
        const type*         a, \
        float16*            b, \
        CUDAContext*        ctx) { \
        LOG(FATAL) << "CUDAContext has not implemented: " \
                   << TypeMetaToString(TypeMeta::Make<type>()) << " -> float16"; \
    }

#define DEFINE_TYPE_ENABLE_FP16_FP32 \
    template <> void TypeA2B<float16, float, CUDAContext>( \
        const int           count, \
        const float16*      a, \
        float*              b, \
        CUDAContext*        ctx) { \
        _TypeHalf2Float \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, reinterpret_cast<const half*>(a), b); \
    } \
    template <> void TypeA2B<float, float16, CUDAContext>( \
        const int           count, \
        const float*        a, \
        float16*            b, \
        CUDAContext*        ctx) { \
        _TypeFloat2Half \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, a, reinterpret_cast<half*>(b)); \
    }

template <> void TypeA2B<float16, float16, CUDAContext>(
    const int               count,
    const float16*          a,
    float16*                b,
    CUDAContext*            ctx) {
    _TypeHalf2Half
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, reinterpret_cast<const half*>(a),
            reinterpret_cast<half*>(b));
}

DEFINE_TYPE_ENABLE_FP16_FP32;
DEFINE_TYPE_DISABLE_FP16(double);
DEFINE_TYPE_DISABLE_FP16(int);
DEFINE_TYPE_DISABLE_FP16(int64_t);
DEFINE_TYPE_DISABLE_FP16(uint8_t);

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA