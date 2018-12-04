#include "core/context.h"
#include "utils/math_functions.h"

namespace dragon {

namespace math {

/******************** Level-0 ********************/

template <> void Set<float16, CPUContext>(
    const int               n,
    const float16           alpha,
    float16*                x,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

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

/******************** Level-1 ********************/

template <> void Add<float16, CPUContext>(
    const int               n,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Sub<float16, CPUContext>(
    const int               n,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Mul<float16, CPUContext>(
    const int               n,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Div<float16, CPUContext>(
    const int               n,
    const float16*          a,
    const float16*          b,
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

template <> void Square<float16, CPUContext>(
    int                     n,
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

template <> void Pow<float16, CPUContext>(
    int                     n,
    const float             alpha,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Inv<float16, CPUContext>(
    const int               n,
    const float             numerator,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** Level-2 ********************/

template <> void Scal<float16, CPUContext>(
    const int               n,
    const float             alpha,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Scale<float16, CPUContext>(
    const int               n,
    const float             alpha,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Dot<float16, CPUContext>(
    int                     n,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void AddScalar<float16, CPUContext>(
    const int               n,
    const float             alpha,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void MulScalar<float16, CPUContext>(
    const int               n,
    const float             alpha,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Axpy<float16, CPUContext>(
    const int               n,
    float                   alpha,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Axpby<float16, CPUContext>(
    const int               n,
    float                   alpha,
    const float16*          x,
    float                   beta,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** Level-3 ********************/

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
 
}  // namespace math

}  // namespace dragon