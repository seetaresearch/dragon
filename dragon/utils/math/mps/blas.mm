#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]]; // alpha
constant float float_arg2 [[function_constant(1)]]; // beta
constant uint uint_arg1 [[function_constant(2)]];   // N
constant uint uint_arg2 [[function_constant(3)]];   // ldx
constant uint uint_arg3 [[function_constant(4)]];   // ldy

template <typename T>
kernel void Scale(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = x[index] * T(float_arg1);
}

template <typename T>
kernel void CopyMatrix(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const uint i = index / uint_arg1, j = index % uint_arg1;
  y[i * uint_arg3 + j] = x[i * uint_arg2 + j];
}

template <typename T>
kernel void Axpy(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] += T(float_arg1) * x[index];
}

template <typename T>
kernel void Axpby(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = T(float_arg1) * x[index] + (T(float_arg2) * y[index]);
}

template<> [[host_name("Axpby_half")]]
kernel void Axpby<half>(
    device const half* x,
    device half* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = fma(half(float_arg1), x[index], half(float_arg2) * y[index]);
}

template<> [[host_name("Axpby_float")]]
kernel void Axpby<float>(
    device const float* x,
    device float* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = fma(float_arg1, x[index], float_arg2 * y[index]);
}

#if defined(__HAVE_NATIVE_DOUBLE__)
template<> [[host_name("Axpby_double")]]
kernel void Axpby<float>(
    device const double* x,
    device double* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = fma(double(float_arg1), x[index], double(float_arg2) * y[index]);
}
#endif

#define INSTANTIATE_UNARY_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device T*, uint);

INSTANTIATE_UNARY_KERNEL(Scale, uint8_t);
INSTANTIATE_UNARY_KERNEL(Scale, int8_t);
INSTANTIATE_UNARY_KERNEL(Scale, int);
INSTANTIATE_UNARY_KERNEL(Scale, int64_t);
INSTANTIATE_UNARY_KERNEL(Scale, half);
INSTANTIATE_UNARY_KERNEL(Scale, float);
INSTANTIATE_UNARY_KERNEL(Axpy, uint8_t);
INSTANTIATE_UNARY_KERNEL(Axpy, int8_t);
INSTANTIATE_UNARY_KERNEL(Axpy, int);
INSTANTIATE_UNARY_KERNEL(Axpy, int64_t);
INSTANTIATE_UNARY_KERNEL(Axpy, half);
INSTANTIATE_UNARY_KERNEL(Axpy, float);
INSTANTIATE_UNARY_KERNEL(Axpby, uint8_t);
INSTANTIATE_UNARY_KERNEL(Axpby, int8_t);
INSTANTIATE_UNARY_KERNEL(Axpby, int);
INSTANTIATE_UNARY_KERNEL(Axpby, int64_t);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_UNARY_KERNEL(Scale, double);
INSTANTIATE_UNARY_KERNEL(Axpy, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_UNARY_KERNEL

#define INSTANTIATE_COPY_MATRIX_KERNEL(T) \
  template [[host_name("CopyMatrix_"#T)]] \
  kernel void CopyMatrix(device const T*, device T*, uint);

INSTANTIATE_COPY_MATRIX_KERNEL(bool);
INSTANTIATE_COPY_MATRIX_KERNEL(uint8_t);
INSTANTIATE_COPY_MATRIX_KERNEL(int8_t);
INSTANTIATE_COPY_MATRIX_KERNEL(int);
INSTANTIATE_COPY_MATRIX_KERNEL(int64_t);
INSTANTIATE_COPY_MATRIX_KERNEL(half);
INSTANTIATE_COPY_MATRIX_KERNEL(float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_COPY_MATRIX_KERNEL(double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)

)";

} // namespace

#define DEFINE_COPY_FUNC(T)                                        \
  template <>                                                      \
  DRAGON_API void Copy<T, MPSContext>(                             \
      const int N, const T* x, T* y, MPSContext* ctx) {            \
    if (N <= 0 || x == y) return;                                  \
    ctx->MemcpyAsync<MPSContext, MPSContext>(N * sizeof(T), y, x); \
  }

DEFINE_COPY_FUNC(bool);
DEFINE_COPY_FUNC(int8_t);
DEFINE_COPY_FUNC(uint8_t);
DEFINE_COPY_FUNC(int);
DEFINE_COPY_FUNC(int64_t);
DEFINE_COPY_FUNC(float16);
DEFINE_COPY_FUNC(float);
DEFINE_COPY_FUNC(double);
#undef DEFINE_COPY_FUNC

#define DEFINE_COPY_FUNC(T)                                                   \
  template <>                                                                 \
  DRAGON_API void Copy<T, MPSContext>(                                        \
      const int N,                                                            \
      const int x_offset,                                                     \
      const int y_offset,                                                     \
      const T* x,                                                             \
      T* y,                                                                   \
      MPSContext* ctx) {                                                      \
    if (N <= 0 || (x == y && x_offset == y_offset)) return;                   \
    auto* encoder = [ctx->mps_stream()->command_buffer() blitCommandEncoder]; \
    [encoder copyFromBuffer:id<MTLBuffer>(x)                                  \
               sourceOffset:sizeof(T) * x_offset                              \
                   toBuffer:id<MTLBuffer>(y)                                  \
          destinationOffset:sizeof(T) * y_offset                              \
                       size:sizeof(T) * N];                                   \
    [encoder endEncoding];                                                    \
    [encoder release];                                                        \
  }

DEFINE_COPY_FUNC(bool);
DEFINE_COPY_FUNC(int8_t);
DEFINE_COPY_FUNC(uint8_t);
DEFINE_COPY_FUNC(int);
DEFINE_COPY_FUNC(int64_t);
DEFINE_COPY_FUNC(float16);
DEFINE_COPY_FUNC(float);
DEFINE_COPY_FUNC(double);
#undef DEFINE_COPY_FUNC

#define DEFINE_COPY_MATRIX_FUNC(T)                                        \
  template <>                                                             \
  DRAGON_API void CopyMatrix<T, MPSContext>(                              \
      const int M,                                                        \
      const int N,                                                        \
      const int ldx,                                                      \
      const int ldy,                                                      \
      const int x_offset,                                                 \
      const int y_offset,                                                 \
      const T* x,                                                         \
      T* y,                                                               \
      MPSContext* ctx) {                                                  \
    if (M <= 0 || N <= 0) return;                                         \
    if (M == 1) {                                                         \
      Copy(N, x_offset, y_offset, x, y, ctx);                             \
      return;                                                             \
    }                                                                     \
    const uint arg1 = N, arg2 = ldx, arg3 = ldy;                          \
    const auto x_bytes_offset = sizeof(T) * x_offset;                     \
    const auto y_bytes_offset = sizeof(T) * y_offset;                     \
    auto kernel = MPSKernel::TypedString<T>("CopyMatrix");                \
    auto args = vector<MPSConstant>(                                      \
        {MPSConstant(&arg1, MTLDataTypeUInt, 2),                          \
         MPSConstant(&arg2, MTLDataTypeUInt, 3),                          \
         MPSConstant(&arg3, MTLDataTypeUInt, 4)});                        \
    auto* command_buffer = ctx->mps_stream()->command_buffer();           \
    auto* encoder = [command_buffer computeCommandEncoder];               \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);     \
    [encoder setComputePipelineState:pso];                                \
    [encoder setBuffer:id<MTLBuffer>(x) offset:x_bytes_offset atIndex:0]; \
    [encoder setBuffer:id<MTLBuffer>(y) offset:y_bytes_offset atIndex:1]; \
    MPSDispatchThreads((M * N), encoder, pso);                            \
    [encoder endEncoding];                                                \
    [encoder release];                                                    \
  }

DEFINE_COPY_MATRIX_FUNC(bool);
DEFINE_COPY_MATRIX_FUNC(int8_t);
DEFINE_COPY_MATRIX_FUNC(uint8_t);
DEFINE_COPY_MATRIX_FUNC(int);
DEFINE_COPY_MATRIX_FUNC(int64_t);
DEFINE_COPY_MATRIX_FUNC(float16);
DEFINE_COPY_MATRIX_FUNC(float);
DEFINE_COPY_MATRIX_FUNC(double);
#undef DEFINE_COPY_MATRIX_FUNC

#define DEFINE_SCALE_FUNC(T)                                                \
  template <>                                                               \
  DRAGON_API void Scale<T, MPSContext>(                                     \
      const int N, const float alpha, const T* x, T* y, MPSContext* ctx) {  \
    if (alpha != 1.f) {                                                     \
      auto kernel = MPSKernel::TypedString<T>("Scale");                     \
      vector<MPSConstant> args({MPSConstant(&alpha, MTLDataTypeFloat, 0)}); \
      auto* command_buffer = ctx->mps_stream()->command_buffer();           \
      auto* encoder = [command_buffer computeCommandEncoder];               \
      auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);     \
      [encoder setComputePipelineState:pso];                                \
      [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];              \
      [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];              \
      MPSDispatchThreads(N, encoder, pso);                                  \
      [encoder endEncoding];                                                \
      [encoder release];                                                    \
      return;                                                               \
    }                                                                       \
    if (x != y) {                                                           \
      ctx->MemcpyAsync<MPSContext, MPSContext>(N * sizeof(T), y, x);        \
    }                                                                       \
  }

DEFINE_SCALE_FUNC(uint8_t);
DEFINE_SCALE_FUNC(int8_t);
DEFINE_SCALE_FUNC(int);
DEFINE_SCALE_FUNC(int64_t);
DEFINE_SCALE_FUNC(float16);
DEFINE_SCALE_FUNC(float);
DEFINE_SCALE_FUNC(double);
#undef DEFINE_SCALE_FUNC

#define DEFINE_AXPY_FUNC(T)                                                \
  template <>                                                              \
  DRAGON_API void Axpy<T, MPSContext>(                                     \
      const int N, const float alpha, const T* x, T* y, MPSContext* ctx) { \
    auto kernel = MPSKernel::TypedString<T>("Axpy");                       \
    vector<MPSConstant> args({MPSConstant(&alpha, MTLDataTypeFloat, 0)});  \
    auto* command_buffer = ctx->mps_stream()->command_buffer();            \
    auto* encoder = [command_buffer computeCommandEncoder];                \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);      \
    [encoder setComputePipelineState:pso];                                 \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];               \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];               \
    MPSDispatchThreads(N, encoder, pso);                                   \
    [encoder endEncoding];                                                 \
    [encoder release];                                                     \
  }

DEFINE_AXPY_FUNC(uint8_t);
DEFINE_AXPY_FUNC(int8_t);
DEFINE_AXPY_FUNC(int);
DEFINE_AXPY_FUNC(int64_t);
DEFINE_AXPY_FUNC(float16);
DEFINE_AXPY_FUNC(float);
DEFINE_AXPY_FUNC(double);
#undef DEFINE_AXPY_FUNC

#define DEFINE_AXPBY_FUNC(T)                                          \
  template <>                                                         \
  DRAGON_API void Axpby<T, MPSContext>(                               \
      const int N,                                                    \
      const float alpha,                                              \
      const T* x,                                                     \
      const float beta,                                               \
      T* y,                                                           \
      MPSContext* ctx) {                                              \
    auto args = vector<MPSConstant>(                                  \
        {MPSConstant(&alpha, MTLDataTypeFloat, 0),                    \
         MPSConstant(&beta, MTLDataTypeFloat, 1)});                   \
    auto kernel = MPSKernel::TypedString<T>("Axpby");                 \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];          \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];          \
    MPSDispatchThreads(N, encoder, pso);                              \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_AXPBY_FUNC(uint8_t);
DEFINE_AXPBY_FUNC(int8_t);
DEFINE_AXPBY_FUNC(int);
DEFINE_AXPBY_FUNC(int64_t);
DEFINE_AXPBY_FUNC(float16);
DEFINE_AXPBY_FUNC(float);
DEFINE_AXPBY_FUNC(double);
#undef DEFINE_AXPBY_FUNC

#define DEFINE_DOT_FUNC(T)                                    \
  template <>                                                 \
  DRAGON_API T Dot<T, MPSContext>(                            \
      const int N, const T* a, const T* b, MPSContext* ctx) { \
    LOG(FATAL) << "DotFunction is not supported.";            \
    return convert::To<T>(0.f);                               \
  }

DEFINE_DOT_FUNC(float16);
DEFINE_DOT_FUNC(float);
DEFINE_DOT_FUNC(double);
#undef DEFINE_DOT_FUNC

} // namespace math

} // namespace dragon
