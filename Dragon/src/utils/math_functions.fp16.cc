#include "core/context.h"
#include "utils/omp_alternative.h"
#include "utils/math_functions.h"

namespace dragon {

namespace math {

/*!
 * ----------------------------------------------
 *
 *
 *            Simple Unary Functions
 *
 *
 * ----------------------------------------------
 */

template <> void Exp<float16, CPUContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Log<float16, CPUContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Inv<float16, CPUContext>(
    const int               n,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Sqrt<float16, CPUContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void RSqrt<float16, CPUContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Square<float16, CPUContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*!
 * ----------------------------------------------
 *
 *
 *             Scale Unary Functions
 *
 *
 * ----------------------------------------------
 */

/*!                y = a                 */

template <> void Set<float16, CPUContext>(
    const int               n,
    const float16           alpha,
    float16*                x,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) x[i] = alpha;
}

/*!                y = x^e                */

template <> void Pow<float16, CPUContext>(
    int                     n,
    const float             alpha,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*!        y = ax    ||    x = ax        */

template <> void Scale<float16, CPUContext>(
    const int               n,
    const float             alpha,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*!                y += ax                */

template <> void Axpy<float16, CPUContext>(
    const int               n,
    float                   alpha,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*!                 y += a                */

template <> void AddScalar<float16, CPUContext>(
    const int               n,
    const float             alpha,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*!
 * ----------------------------------------------
 *
 *
 *             Extended Unary Functions
 *
 *
 * ----------------------------------------------
 */

template <> void InvStd<float16, CPUContext>(
    const int               n,
    const float             eps,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*!                y = sum(x)               */

template <> void Sum<float16, CPUContext>(
    const int               n,
    const float             alpha,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*!
 * ----------------------------------------------
 *
 *
 *            Simply Binary Functions
 *
 *
 * ----------------------------------------------
 */

#define DEFINE_SIMPLE_BINARY_FUNC(name) \
    template <> void name<float16, CPUContext>( \
        const int               n, \
        const float16*          a, \
        const float16*          b, \
        float16*                y, \
        CPUContext*             ctx) { \
        CPU_FP16_NOT_SUPPORTED; \
    }

DEFINE_SIMPLE_BINARY_FUNC(Add);
DEFINE_SIMPLE_BINARY_FUNC(Sub);
DEFINE_SIMPLE_BINARY_FUNC(Mul);
DEFINE_SIMPLE_BINARY_FUNC(Div);
#undef DEFINE_SIMPLE_BINARY_FUNC

template <> void Dot<float16, CPUContext>(
    int                     n,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*!
 * ----------------------------------------------
 *
 *
 *          Broadcast Binary Functions
 *
 *
 * ----------------------------------------------
 */

#define DEFINE_BROADCAST_BINARY_FUNC(name) \
    template <> void Broadcast##name<float16, CPUContext>( \
        const int               rows, \
        const int               cols, \
        const int               type, \
        const float16*          a, \
        const float16*          b, \
        float16*                y, \
        CPUContext*             ctx) { \
        CPU_FP16_NOT_SUPPORTED; \
    }

DEFINE_BROADCAST_BINARY_FUNC(Add);
DEFINE_BROADCAST_BINARY_FUNC(Sub);
DEFINE_BROADCAST_BINARY_FUNC(Mul);
DEFINE_BROADCAST_BINARY_FUNC(Div);
#undef DEFINE_BROADCAST_BINARY_FUNC

/*!
 * ----------------------------------------------
 *
 *
 *        Linear Algebra Binary Functions
 *
 *
 * ----------------------------------------------
 */

template <> void Gemm<float16, CPUContext>(
    const CBLAS_TRANSPOSE   TransA,
    const CBLAS_TRANSPOSE   TransB,
    const int               M,
    const int               N,
    const int               K,
    const float             alpha,
    const float16*          A,
    const float16*          B,
    const float             beta,
    float16*                C,
    CPUContext*             ctx,
    TensorProto_DataType    math_type) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Gemv<float16, CPUContext>(
    const CBLAS_TRANSPOSE   TransA,
    const int               M,
    const int               N,
    const float             alpha,
    const float16*          A,
    const float16*          x,
    const float             beta,
    float16*                y,
    CPUContext*             ctx,
    TensorProto_DataType    math_type) {
    CPU_FP16_NOT_SUPPORTED;
}

/*!
 * ----------------------------------------------
 *
 *
 *               Random Functions
 *
 *
 * ----------------------------------------------
 */

template <> void RandomUniform<float16, CPUContext>(
    const int               n,
    const float             low,
    const float             high,
    float16*                x,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void RandomNormal<float16, CPUContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    float16*                x,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void RandomTruncatedNormal<float16, CPUContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    const float             low,
    const float             high,
    float16*                x,
    CPUContext*             ctx) {
    NOT_IMPLEMENTED;
}

}  // namespace math

}  // namespace dragon