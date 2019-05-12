#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Astype <Ta = ?, Tb = ?, Device = CUDA> */

template <typename Ta, typename Tb>
__global__ void _TypeA2B(
    const int               nthreads,
    const Ta*               a,
    Tb*                     b) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        b[i] = a[i];
    }
}

#define DEFINE_TYPE_A_TO_B(Ta, Tb) \
    template <> void TypeA2B<Ta, Tb, CUDAContext>( \
        const int           count, \
        const Ta*           a, \
        Tb*                 b, \
        CUDAContext*        ctx) { \
        _TypeA2B \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, a, b \
        ); \
    }

#define DEFINE_TYPE_A_TO_ALL(type_a) \
    DEFINE_TYPE_A_TO_B(type_a, bool); \
    DEFINE_TYPE_A_TO_B(type_a, int8_t); \
    DEFINE_TYPE_A_TO_B(type_a, uint8_t); \
    DEFINE_TYPE_A_TO_B(type_a, int); \
    DEFINE_TYPE_A_TO_B(type_a, int64_t); \
    DEFINE_TYPE_A_TO_B(type_a, float); \
    DEFINE_TYPE_A_TO_B(type_a, double);

DEFINE_TYPE_A_TO_ALL(bool);
DEFINE_TYPE_A_TO_ALL(int8_t);
DEFINE_TYPE_A_TO_ALL(uint8_t);
DEFINE_TYPE_A_TO_ALL(int);
DEFINE_TYPE_A_TO_ALL(int64_t);
DEFINE_TYPE_A_TO_ALL(float);
DEFINE_TYPE_A_TO_ALL(double);

/*! Astype <Ta = float16, Tb = float32, Device = CUDA> */

template<> __global__ void _TypeA2B<half, float>(
    const int               nthreads,
    const half*             a,
    float*                  b) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        b[i] = __half2float(a[i]);
    }
}

template <> void TypeA2B<float16, float, CUDAContext>(
    const int               count,
    const float16*          a,
    float*                  b, 
    CUDAContext*            ctx) {
    _TypeA2B
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count, reinterpret_cast<const half*>(a), b
    );
}

/*! Astype <Ta = float32, Tb = float16, Device = CUDA> */

template<> __global__ void _TypeA2B<float, half>(
    const int               nthreads,
    const float*            a,
    half*                   b) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        b[i] = __float2half(a[i]);
    }
}

template <> void TypeA2B<float, float16, CUDAContext>(
    const int           count,
    const float*        a,
    float16*            b,
    CUDAContext*        ctx) {
    _TypeA2B
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count, a, reinterpret_cast<half*>(b)
    );
}

/*! Astype <Ta = float16, Tb = float16, Device = CUDA> */

template<> __global__ void _TypeA2B<half, half>(
    const int               nthreads,
    const half*             a,
    half*                   b) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        b[i] = a[i];
    }
}

template <> void TypeA2B<float16, float16, CUDAContext>(
    const int               count,
    const float16*          a,
    float16*                b,
    CUDAContext*            ctx) {
    _TypeA2B
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count,
        reinterpret_cast<const half*>(a),
        reinterpret_cast<half*>(b)
    );
}

#define DEFINE_TYPE_FP16_DISABLED(T) \
    template <> void TypeA2B<float16, T, CUDAContext>( \
        const int           count, \
        const float16*      a, \
        T*                  b, \
        CUDAContext*        ctx) { \
        LOG(FATAL) << "Not Implemented: float16 -> " \
                   << TypeMetaToString(TypeMeta::Make<T>()); \
    } \
    template <> void TypeA2B<T, float16, CUDAContext>( \
        const int           count, \
        const T*            a, \
        float16*            b, \
        CUDAContext*        ctx) { \
        LOG(FATAL) << "Not Implemented: " \
                   << TypeMetaToString(TypeMeta::Make<T>()) \
                   << " -> float16"; \
    }

DEFINE_TYPE_FP16_DISABLED(bool);
DEFINE_TYPE_FP16_DISABLED(int8_t);
DEFINE_TYPE_FP16_DISABLED(uint8_t);
DEFINE_TYPE_FP16_DISABLED(int);
DEFINE_TYPE_FP16_DISABLED(int64_t);
DEFINE_TYPE_FP16_DISABLED(double);

#undef DEFINE_TYPE_A_TO_B
#undef DEFINE_TYPE_A_TO_ALL
#undef DEFINE_TYPE_FP16_DISABLED

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA