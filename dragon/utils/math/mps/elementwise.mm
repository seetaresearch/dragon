#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]];

#define DEFINE_UNARY_KERNEL(name, func)               \
  template <typename InputT, typename OutputT>        \
  kernel void name(                                   \
      device const InputT* x,                         \
      device OutputT* y,                              \
      const uint index [[thread_position_in_grid]]) { \
    y[index] = func(x[index]);                        \
  }

template <typename T>
kernel void Square(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const T val = x[index];
  y[index] = val * val;
}

template <typename T>
kernel void Bias(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = x[index] + T(float_arg1);
}

template <typename T>
kernel void Inv(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = T(1) / x[index];
}

template <typename T>
kernel void InvStd(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = rsqrt(x[index] + T(float_arg1));
}

template <typename T>
kernel void Sign(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = T(sign(float(x[index])));
}

template <typename T>
kernel void NaNToNum(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const T val = x[index];
  y[index] = isnan(val) ? T(float_arg1) : val;
}

DEFINE_UNARY_KERNEL(Not, !);
DEFINE_UNARY_KERNEL(BitwiseNot, ~);
DEFINE_UNARY_KERNEL(Neg, -);
DEFINE_UNARY_KERNEL(Abs, abs);
DEFINE_UNARY_KERNEL(Ceil, ceil);
DEFINE_UNARY_KERNEL(Floor, floor);
DEFINE_UNARY_KERNEL(Round, round);
DEFINE_UNARY_KERNEL(Exp, exp);
DEFINE_UNARY_KERNEL(Log, log);
DEFINE_UNARY_KERNEL(Sqrt, sqrt);
DEFINE_UNARY_KERNEL(Rsqrt, rsqrt);
DEFINE_UNARY_KERNEL(Sin, sin);
DEFINE_UNARY_KERNEL(Cos, cos);
DEFINE_UNARY_KERNEL(IsInf, isinf);
DEFINE_UNARY_KERNEL(IsNaN, isnan);
DEFINE_UNARY_KERNEL(IsFinite, isfinite);
#undef DEFINE_UNARY_KERNEL

#define INSTANTIATE_UNARY_KERNEL(name, InputT, OutputT) \
  template [[host_name(#name"_"#InputT)]] \
  kernel void name(device const InputT*, device OutputT*, uint);

INSTANTIATE_UNARY_KERNEL(Neg, int8_t, int8_t);
INSTANTIATE_UNARY_KERNEL(Neg, int, int);
INSTANTIATE_UNARY_KERNEL(Neg, int64_t, int64_t);
INSTANTIATE_UNARY_KERNEL(Neg, half, half);
INSTANTIATE_UNARY_KERNEL(Neg, float, float);
INSTANTIATE_UNARY_KERNEL(Abs, int8_t, int8_t);
INSTANTIATE_UNARY_KERNEL(Abs, int, int);
INSTANTIATE_UNARY_KERNEL(Abs, int64_t, int64_t);
INSTANTIATE_UNARY_KERNEL(Abs, half, half);
INSTANTIATE_UNARY_KERNEL(Abs, float, float);
INSTANTIATE_UNARY_KERNEL(Square, uint8_t, uint8_t);
INSTANTIATE_UNARY_KERNEL(Square, int8_t, int8_t);
INSTANTIATE_UNARY_KERNEL(Square, int, int);
INSTANTIATE_UNARY_KERNEL(Square, int64_t, int64_t);
INSTANTIATE_UNARY_KERNEL(Square, half, half);
INSTANTIATE_UNARY_KERNEL(Square, float, float);
INSTANTIATE_UNARY_KERNEL(Bias, uint8_t, uint8_t);
INSTANTIATE_UNARY_KERNEL(Bias, int8_t, int8_t);
INSTANTIATE_UNARY_KERNEL(Bias, int, int);
INSTANTIATE_UNARY_KERNEL(Bias, int64_t, int64_t);
INSTANTIATE_UNARY_KERNEL(Bias, half, half);
INSTANTIATE_UNARY_KERNEL(Bias, float, float);
INSTANTIATE_UNARY_KERNEL(Sign, uint8_t, uint8_t);
INSTANTIATE_UNARY_KERNEL(Sign, int8_t, int8_t);
INSTANTIATE_UNARY_KERNEL(Sign, int, int);
INSTANTIATE_UNARY_KERNEL(Sign, int64_t, int64_t);
INSTANTIATE_UNARY_KERNEL(Sign, half, half);
INSTANTIATE_UNARY_KERNEL(Sign, float, float);
INSTANTIATE_UNARY_KERNEL(Ceil, half, half);
INSTANTIATE_UNARY_KERNEL(Ceil, float, float);
INSTANTIATE_UNARY_KERNEL(Floor, half, half);
INSTANTIATE_UNARY_KERNEL(Floor, float, float);
INSTANTIATE_UNARY_KERNEL(Round, half, half);
INSTANTIATE_UNARY_KERNEL(Round, float, float);
INSTANTIATE_UNARY_KERNEL(Exp, half, half);
INSTANTIATE_UNARY_KERNEL(Exp, float, float);
INSTANTIATE_UNARY_KERNEL(Log, half, half);
INSTANTIATE_UNARY_KERNEL(Log, float, float);
INSTANTIATE_UNARY_KERNEL(Inv, half, half);
INSTANTIATE_UNARY_KERNEL(Inv, float, float);
INSTANTIATE_UNARY_KERNEL(InvStd, half, half);
INSTANTIATE_UNARY_KERNEL(InvStd, float, float);
INSTANTIATE_UNARY_KERNEL(Sqrt, half, half);
INSTANTIATE_UNARY_KERNEL(Sqrt, float, float);
INSTANTIATE_UNARY_KERNEL(Rsqrt, half, half);
INSTANTIATE_UNARY_KERNEL(Rsqrt, float, float);
INSTANTIATE_UNARY_KERNEL(Sin, half, half);
INSTANTIATE_UNARY_KERNEL(Sin, float, float);
INSTANTIATE_UNARY_KERNEL(Cos, half, half);
INSTANTIATE_UNARY_KERNEL(Cos, float, float);
INSTANTIATE_UNARY_KERNEL(NaNToNum, half, half);
INSTANTIATE_UNARY_KERNEL(NaNToNum, float, float);
INSTANTIATE_UNARY_KERNEL(Not, bool, bool);
INSTANTIATE_UNARY_KERNEL(Not, uint8_t, bool);
INSTANTIATE_UNARY_KERNEL(Not, int8_t, bool);
INSTANTIATE_UNARY_KERNEL(Not, int, bool);
INSTANTIATE_UNARY_KERNEL(Not, int64_t, bool);
INSTANTIATE_UNARY_KERNEL(Not, half, bool);
INSTANTIATE_UNARY_KERNEL(Not, float, bool);
INSTANTIATE_UNARY_KERNEL(BitwiseNot, uint8_t, uint8_t);
INSTANTIATE_UNARY_KERNEL(BitwiseNot, int8_t, int8_t);
INSTANTIATE_UNARY_KERNEL(BitwiseNot, int, int);
INSTANTIATE_UNARY_KERNEL(BitwiseNot, int64_t, int64_t);
INSTANTIATE_UNARY_KERNEL(IsInf, half, bool);
INSTANTIATE_UNARY_KERNEL(IsInf, float, bool);
INSTANTIATE_UNARY_KERNEL(IsNaN, half, bool);
INSTANTIATE_UNARY_KERNEL(IsNaN, float, bool);
INSTANTIATE_UNARY_KERNEL(IsFinite, half, bool);
INSTANTIATE_UNARY_KERNEL(IsFinite, float, bool);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_UNARY_KERNEL(Neg, double, double);
INSTANTIATE_UNARY_KERNEL(Abs, double, double);
INSTANTIATE_UNARY_KERNEL(Square, double, double);
INSTANTIATE_UNARY_KERNEL(Bias, double, double);
INSTANTIATE_UNARY_KERNEL(Sign, double, double);
INSTANTIATE_UNARY_KERNEL(Ceil, double, double);
INSTANTIATE_UNARY_KERNEL(Floor, double, double);
INSTANTIATE_UNARY_KERNEL(Round, double, double);
INSTANTIATE_UNARY_KERNEL(Exp, double, double);
INSTANTIATE_UNARY_KERNEL(Log, double, double);
INSTANTIATE_UNARY_KERNEL(Inv, double, double);
INSTANTIATE_UNARY_KERNEL(InvStd, double, double);
INSTANTIATE_UNARY_KERNEL(Sqrt, double, double);
INSTANTIATE_UNARY_KERNEL(Rsqrt, double, double);
INSTANTIATE_UNARY_KERNEL(Sin, double, double);
INSTANTIATE_UNARY_KERNEL(Cos, double, double);
INSTANTIATE_UNARY_KERNEL(NaNToNum, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_UNARY_KERNEL

template <typename T>
kernel void Set(
    device T* x,
    const uint index [[thread_position_in_grid]]) {
  x[index] = T(float_arg1);
}

template [[host_name("Set_bool")]] kernel void Set(device bool*, uint);
template [[host_name("Set_uint8_t")]] kernel void Set(device uint8_t*, uint);
template [[host_name("Set_int8_t")]] kernel void Set(device int8_t*, uint);
template [[host_name("Set_int")]] kernel void Set(device int*, uint);
template [[host_name("Set_int64_t")]] kernel void Set(device int64_t*, uint);
template [[host_name("Set_half")]] kernel void Set(device half*, uint);
template [[host_name("Set_float")]] kernel void Set(device float*, uint);
#if defined(__HAVE_NATIVE_DOUBLE__)
template [[host_name("Set_double")]] kernel void Set(device double*, uint);
#endif // defined(__HAVE_NATIVE_DOUBLE__)

template <typename T>
kernel void ApplyMask(
    device const uint8_t* mask,
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = x[index] * T(float(mask[index]) * float_arg1);
}

#define INSTANTIATE_UNARY_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const uint8_t*, device const T*, device T*, uint);

INSTANTIATE_UNARY_KERNEL(ApplyMask, uint8_t);
INSTANTIATE_UNARY_KERNEL(ApplyMask, int8_t);
INSTANTIATE_UNARY_KERNEL(ApplyMask, int);
INSTANTIATE_UNARY_KERNEL(ApplyMask, int64_t);
INSTANTIATE_UNARY_KERNEL(ApplyMask, half);
INSTANTIATE_UNARY_KERNEL(ApplyMask, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_UNARY_KERNEL(ApplyMask, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_UNARY_KERNEL

#define DEFINE_BINARY_KERNEL(name, expr)              \
  template <typename InputT, typename OutputT>        \
  kernel void name(                                   \
      device const InputT* a,                         \
      device const InputT* b,                         \
      device OutputT* y,                              \
      const uint index [[thread_position_in_grid]]) { \
    y[index] = a[index] expr b[index];                \
  }

DEFINE_BINARY_KERNEL(Add, +);
DEFINE_BINARY_KERNEL(Sub, -);
DEFINE_BINARY_KERNEL(Mul, *);
DEFINE_BINARY_KERNEL(Div, /);
DEFINE_BINARY_KERNEL(BitwiseAnd, &);
DEFINE_BINARY_KERNEL(BitwiseOr, |);
DEFINE_BINARY_KERNEL(BitwiseXor, ^);
DEFINE_BINARY_KERNEL(And, &&);
DEFINE_BINARY_KERNEL(Or, ||);
DEFINE_BINARY_KERNEL(Equal, ==);
DEFINE_BINARY_KERNEL(NotEqual, !=);
DEFINE_BINARY_KERNEL(Less, <);
DEFINE_BINARY_KERNEL(LessEqual, <=);
DEFINE_BINARY_KERNEL(Greater, >);
DEFINE_BINARY_KERNEL(GreaterEqual, >=);
#undef DEFINE_BINARY_KERNEL

#define DEFINE_BINARY_KERNEL(name, func)              \
  template <typename InputT, typename OutputT>        \
  kernel void name(                                   \
      device const InputT* a,                         \
      device const InputT* b,                         \
      device OutputT* y,                              \
      const uint index [[thread_position_in_grid]]) { \
    y[index] = func(a[index], b[index]);              \
  }

DEFINE_BINARY_KERNEL(Pow, pow);
DEFINE_BINARY_KERNEL(Atan2, atan2);
DEFINE_BINARY_KERNEL(Minimum, min);
DEFINE_BINARY_KERNEL(Maximum, max);
#undef DEFINE_BINARY_KERNEL

template <typename InputT, typename OutputT>
kernel void Xor(
    device const InputT* a,
    device const InputT* b,
    device OutputT* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = bool(a[index]) ^ bool(b[index]);
}

#define INSTANTIATE_BINARY_KERNEL(name, InputT, OutputT) \
  template [[host_name(#name"_"#InputT)]] \
  kernel void name(device const InputT*, device const InputT*, \
                   device OutputT*, uint);

INSTANTIATE_BINARY_KERNEL(Add, uint8_t, uint8_t);
INSTANTIATE_BINARY_KERNEL(Add, int8_t, int8_t);
INSTANTIATE_BINARY_KERNEL(Add, int, int);
INSTANTIATE_BINARY_KERNEL(Add, int64_t, int64_t);
INSTANTIATE_BINARY_KERNEL(Add, half, half);
INSTANTIATE_BINARY_KERNEL(Add, float, float);
INSTANTIATE_BINARY_KERNEL(Sub, uint8_t, uint8_t);
INSTANTIATE_BINARY_KERNEL(Sub, int8_t, int8_t);
INSTANTIATE_BINARY_KERNEL(Sub, int, int);
INSTANTIATE_BINARY_KERNEL(Sub, int64_t, int64_t);
INSTANTIATE_BINARY_KERNEL(Sub, half, half);
INSTANTIATE_BINARY_KERNEL(Sub, float, float);
INSTANTIATE_BINARY_KERNEL(Mul, uint8_t, uint8_t);
INSTANTIATE_BINARY_KERNEL(Mul, int8_t, int8_t);
INSTANTIATE_BINARY_KERNEL(Mul, int, int);
INSTANTIATE_BINARY_KERNEL(Mul, int64_t, int64_t);
INSTANTIATE_BINARY_KERNEL(Mul, half, half);
INSTANTIATE_BINARY_KERNEL(Mul, float, float);
INSTANTIATE_BINARY_KERNEL(Div, uint8_t, uint8_t);
INSTANTIATE_BINARY_KERNEL(Div, int8_t, int8_t);
INSTANTIATE_BINARY_KERNEL(Div, int, int);
INSTANTIATE_BINARY_KERNEL(Div, int64_t, int64_t);
INSTANTIATE_BINARY_KERNEL(Div, half, half);
INSTANTIATE_BINARY_KERNEL(Div, float, float);
INSTANTIATE_BINARY_KERNEL(Minimum, uint8_t, uint8_t);
INSTANTIATE_BINARY_KERNEL(Minimum, int8_t, int8_t);
INSTANTIATE_BINARY_KERNEL(Minimum, int, int);
INSTANTIATE_BINARY_KERNEL(Minimum, int64_t, int64_t);
INSTANTIATE_BINARY_KERNEL(Minimum, half, half);
INSTANTIATE_BINARY_KERNEL(Minimum, float, float);
INSTANTIATE_BINARY_KERNEL(Maximum, uint8_t, uint8_t);
INSTANTIATE_BINARY_KERNEL(Maximum, int8_t, int8_t);
INSTANTIATE_BINARY_KERNEL(Maximum, int, int);
INSTANTIATE_BINARY_KERNEL(Maximum, int64_t, int64_t);
INSTANTIATE_BINARY_KERNEL(Maximum, half, half);
INSTANTIATE_BINARY_KERNEL(Maximum, float, float);
INSTANTIATE_BINARY_KERNEL(Pow, half, half);
INSTANTIATE_BINARY_KERNEL(Pow, float, float);
INSTANTIATE_BINARY_KERNEL(Atan2, half, half);
INSTANTIATE_BINARY_KERNEL(Atan2, float, float);
INSTANTIATE_BINARY_KERNEL(BitwiseAnd, bool, bool);
INSTANTIATE_BINARY_KERNEL(BitwiseAnd, uint8_t, uint8_t);
INSTANTIATE_BINARY_KERNEL(BitwiseAnd, int8_t, int8_t);
INSTANTIATE_BINARY_KERNEL(BitwiseAnd, int, int);
INSTANTIATE_BINARY_KERNEL(BitwiseAnd, int64_t, int64_t);
INSTANTIATE_BINARY_KERNEL(BitwiseOr, bool, bool);
INSTANTIATE_BINARY_KERNEL(BitwiseOr, uint8_t, uint8_t);
INSTANTIATE_BINARY_KERNEL(BitwiseOr, int8_t, int8_t);
INSTANTIATE_BINARY_KERNEL(BitwiseOr, int, int);
INSTANTIATE_BINARY_KERNEL(BitwiseOr, int64_t, int64_t);
INSTANTIATE_BINARY_KERNEL(BitwiseXor, bool, bool);
INSTANTIATE_BINARY_KERNEL(BitwiseXor, uint8_t, uint8_t);
INSTANTIATE_BINARY_KERNEL(BitwiseXor, int8_t, int8_t);
INSTANTIATE_BINARY_KERNEL(BitwiseXor, int, int);
INSTANTIATE_BINARY_KERNEL(BitwiseXor, int64_t, int64_t);
INSTANTIATE_BINARY_KERNEL(And, bool, bool);
INSTANTIATE_BINARY_KERNEL(And, uint8_t, bool);
INSTANTIATE_BINARY_KERNEL(And, int8_t, bool);
INSTANTIATE_BINARY_KERNEL(And, int, bool);
INSTANTIATE_BINARY_KERNEL(And, int64_t, bool);
INSTANTIATE_BINARY_KERNEL(And, half, bool);
INSTANTIATE_BINARY_KERNEL(And, float, bool);
INSTANTIATE_BINARY_KERNEL(Or, bool, bool);
INSTANTIATE_BINARY_KERNEL(Or, uint8_t, bool);
INSTANTIATE_BINARY_KERNEL(Or, int8_t, bool);
INSTANTIATE_BINARY_KERNEL(Or, int, bool);
INSTANTIATE_BINARY_KERNEL(Or, int64_t, bool);
INSTANTIATE_BINARY_KERNEL(Or, half, bool);
INSTANTIATE_BINARY_KERNEL(Or, float, bool);
INSTANTIATE_BINARY_KERNEL(Xor, bool, bool);
INSTANTIATE_BINARY_KERNEL(Xor, uint8_t, bool);
INSTANTIATE_BINARY_KERNEL(Xor, int8_t, bool);
INSTANTIATE_BINARY_KERNEL(Xor, int, bool);
INSTANTIATE_BINARY_KERNEL(Xor, int64_t, bool);
INSTANTIATE_BINARY_KERNEL(Xor, half, bool);
INSTANTIATE_BINARY_KERNEL(Xor, float, bool);
INSTANTIATE_BINARY_KERNEL(Equal, bool, bool);
INSTANTIATE_BINARY_KERNEL(Equal, uint8_t, bool);
INSTANTIATE_BINARY_KERNEL(Equal, int8_t, bool);
INSTANTIATE_BINARY_KERNEL(Equal, int, bool);
INSTANTIATE_BINARY_KERNEL(Equal, int64_t, bool);
INSTANTIATE_BINARY_KERNEL(Equal, half, bool);
INSTANTIATE_BINARY_KERNEL(Equal, float, bool);
INSTANTIATE_BINARY_KERNEL(NotEqual, bool, bool);
INSTANTIATE_BINARY_KERNEL(NotEqual, uint8_t, bool);
INSTANTIATE_BINARY_KERNEL(NotEqual, int8_t, bool);
INSTANTIATE_BINARY_KERNEL(NotEqual, int, bool);
INSTANTIATE_BINARY_KERNEL(NotEqual, int64_t, bool);
INSTANTIATE_BINARY_KERNEL(NotEqual, half, bool);
INSTANTIATE_BINARY_KERNEL(NotEqual, float, bool);
INSTANTIATE_BINARY_KERNEL(Less, bool, bool);
INSTANTIATE_BINARY_KERNEL(Less, uint8_t, bool);
INSTANTIATE_BINARY_KERNEL(Less, int8_t, bool);
INSTANTIATE_BINARY_KERNEL(Less, int, bool);
INSTANTIATE_BINARY_KERNEL(Less, int64_t, bool);
INSTANTIATE_BINARY_KERNEL(Less, half, bool);
INSTANTIATE_BINARY_KERNEL(Less, float, bool);
INSTANTIATE_BINARY_KERNEL(LessEqual, bool, bool);
INSTANTIATE_BINARY_KERNEL(LessEqual, uint8_t, bool);
INSTANTIATE_BINARY_KERNEL(LessEqual, int8_t, bool);
INSTANTIATE_BINARY_KERNEL(LessEqual, int, bool);
INSTANTIATE_BINARY_KERNEL(LessEqual, int64_t, bool);
INSTANTIATE_BINARY_KERNEL(LessEqual, half, bool);
INSTANTIATE_BINARY_KERNEL(LessEqual, float, bool);
INSTANTIATE_BINARY_KERNEL(Greater, bool, bool);
INSTANTIATE_BINARY_KERNEL(Greater, uint8_t, bool);
INSTANTIATE_BINARY_KERNEL(Greater, int8_t, bool);
INSTANTIATE_BINARY_KERNEL(Greater, int, bool);
INSTANTIATE_BINARY_KERNEL(Greater, int64_t, bool);
INSTANTIATE_BINARY_KERNEL(Greater, half, bool);
INSTANTIATE_BINARY_KERNEL(Greater, float, bool);
INSTANTIATE_BINARY_KERNEL(GreaterEqual, bool, bool);
INSTANTIATE_BINARY_KERNEL(GreaterEqual, uint8_t, bool);
INSTANTIATE_BINARY_KERNEL(GreaterEqual, int8_t, bool);
INSTANTIATE_BINARY_KERNEL(GreaterEqual, int, bool);
INSTANTIATE_BINARY_KERNEL(GreaterEqual, int64_t, bool);
INSTANTIATE_BINARY_KERNEL(GreaterEqual, half, bool);
INSTANTIATE_BINARY_KERNEL(GreaterEqual, float, bool);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_BINARY_KERNEL(Add, double, double);
INSTANTIATE_BINARY_KERNEL(Sub, double, double);
INSTANTIATE_BINARY_KERNEL(Mul, double, double);
INSTANTIATE_BINARY_KERNEL(Div, double, double);
INSTANTIATE_BINARY_KERNEL(Minimum, double, double);
INSTANTIATE_BINARY_KERNEL(Maximum, double, double);
INSTANTIATE_BINARY_KERNEL(Pow, double, double);
INSTANTIATE_BINARY_KERNEL(Atan2, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_BINARY_KERNEL

)";

} // namespace

#define DEFINE_UNARY_FUNC(name, InputT, OutputT)                     \
  template <>                                                        \
  DRAGON_API void name<InputT, MPSContext>(                          \
      const int N, const InputT* x, OutputT* y, MPSContext* ctx) {   \
    auto kernel_name = MPSKernel::TypedString<InputT>(#name);        \
    auto* command_buffer = ctx->mps_stream()->command_buffer();      \
    auto* encoder = [command_buffer computeCommandEncoder];          \
    auto* pso = MPSKernel(kernel_name, METAL_SHADERS).GetState(ctx); \
    [encoder setComputePipelineState:pso];                           \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];         \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];         \
    MPSDispatchThreads(N, encoder, pso);                             \
    [encoder endEncoding];                                           \
    [encoder release];                                               \
  }

DEFINE_UNARY_FUNC(Neg, int8_t, int8_t);
DEFINE_UNARY_FUNC(Neg, int, int);
DEFINE_UNARY_FUNC(Neg, int64_t, int64_t);
DEFINE_UNARY_FUNC(Neg, float16, float16);
DEFINE_UNARY_FUNC(Neg, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Neg, float, float);
DEFINE_UNARY_FUNC(Neg, double, double);
DEFINE_UNARY_FUNC(Abs, int8_t, int8_t);
DEFINE_UNARY_FUNC(Abs, int, int);
DEFINE_UNARY_FUNC(Abs, int64_t, int64_t);
DEFINE_UNARY_FUNC(Abs, float16, float16);
DEFINE_UNARY_FUNC(Abs, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Abs, float, float);
DEFINE_UNARY_FUNC(Abs, double, double);
DEFINE_UNARY_FUNC(Square, uint8_t, uint8_t);
DEFINE_UNARY_FUNC(Square, int8_t, int8_t);
DEFINE_UNARY_FUNC(Square, int, int);
DEFINE_UNARY_FUNC(Square, int64_t, int64_t);
DEFINE_UNARY_FUNC(Square, float16, float16);
DEFINE_UNARY_FUNC(Square, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Square, float, float);
DEFINE_UNARY_FUNC(Square, double, double);
DEFINE_UNARY_FUNC(Sign, uint8_t, uint8_t);
DEFINE_UNARY_FUNC(Sign, int8_t, int8_t);
DEFINE_UNARY_FUNC(Sign, int, int);
DEFINE_UNARY_FUNC(Sign, int64_t, int64_t);
DEFINE_UNARY_FUNC(Sign, float16, float16);
DEFINE_UNARY_FUNC(Sign, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Sign, float, float);
DEFINE_UNARY_FUNC(Sign, double, double);
DEFINE_UNARY_FUNC(Ceil, float16, float16);
DEFINE_UNARY_FUNC(Ceil, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Ceil, float, float);
DEFINE_UNARY_FUNC(Ceil, double, double);
DEFINE_UNARY_FUNC(Floor, float16, float16);
DEFINE_UNARY_FUNC(Floor, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Floor, float, float);
DEFINE_UNARY_FUNC(Floor, double, double);
DEFINE_UNARY_FUNC(Round, float16, float16);
DEFINE_UNARY_FUNC(Round, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Round, float, float);
DEFINE_UNARY_FUNC(Round, double, double);
DEFINE_UNARY_FUNC(Exp, float16, float16);
DEFINE_UNARY_FUNC(Exp, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Exp, float, float);
DEFINE_UNARY_FUNC(Exp, double, double);
DEFINE_UNARY_FUNC(Log, float16, float16);
DEFINE_UNARY_FUNC(Log, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Log, float, float);
DEFINE_UNARY_FUNC(Log, double, double);
DEFINE_UNARY_FUNC(Inv, float16, float16);
DEFINE_UNARY_FUNC(Inv, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Inv, float, float);
DEFINE_UNARY_FUNC(Inv, double, double);
DEFINE_UNARY_FUNC(Sqrt, float16, float16);
DEFINE_UNARY_FUNC(Sqrt, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Sqrt, float, float);
DEFINE_UNARY_FUNC(Sqrt, double, double);
DEFINE_UNARY_FUNC(Rsqrt, float16, float16);
DEFINE_UNARY_FUNC(Rsqrt, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Rsqrt, float, float);
DEFINE_UNARY_FUNC(Rsqrt, double, double);
DEFINE_UNARY_FUNC(Sin, float16, float16);
DEFINE_UNARY_FUNC(Sin, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Sin, float, float);
DEFINE_UNARY_FUNC(Sin, double, double);
DEFINE_UNARY_FUNC(Cos, float16, float16);
DEFINE_UNARY_FUNC(Cos, bfloat16, bfloat16);
DEFINE_UNARY_FUNC(Cos, float, float);
DEFINE_UNARY_FUNC(Cos, double, double);
DEFINE_UNARY_FUNC(Not, bool, bool);
DEFINE_UNARY_FUNC(Not, uint8_t, bool);
DEFINE_UNARY_FUNC(Not, int8_t, bool);
DEFINE_UNARY_FUNC(Not, int, bool);
DEFINE_UNARY_FUNC(Not, int64_t, bool);
DEFINE_UNARY_FUNC(Not, float16, bool);
DEFINE_UNARY_FUNC(Not, bfloat16, bool);
DEFINE_UNARY_FUNC(Not, float, bool);
DEFINE_UNARY_FUNC(Not, double, bool);
DEFINE_UNARY_FUNC(BitwiseNot, uint8_t, uint8_t);
DEFINE_UNARY_FUNC(BitwiseNot, int8_t, int8_t);
DEFINE_UNARY_FUNC(BitwiseNot, int, int);
DEFINE_UNARY_FUNC(BitwiseNot, int64_t, int64_t);
DEFINE_UNARY_FUNC(IsInf, float16, bool);
DEFINE_UNARY_FUNC(IsInf, bfloat16, bool);
DEFINE_UNARY_FUNC(IsInf, float, bool);
DEFINE_UNARY_FUNC(IsInf, double, bool);
DEFINE_UNARY_FUNC(IsNaN, float16, bool);
DEFINE_UNARY_FUNC(IsNaN, bfloat16, bool);
DEFINE_UNARY_FUNC(IsNaN, float, bool);
DEFINE_UNARY_FUNC(IsNaN, double, bool);
DEFINE_UNARY_FUNC(IsFinite, float16, bool);
DEFINE_UNARY_FUNC(IsFinite, bfloat16, bool);
DEFINE_UNARY_FUNC(IsFinite, float, bool);
DEFINE_UNARY_FUNC(IsFinite, double, bool);
#undef DEFINE_UNARY_FUNC

template <>
DRAGON_API void BitwiseNot<bool, MPSContext>(
    const int N,
    const bool* x,
    bool* y,
    MPSContext* ctx) {
  math::Not(N, x, y, ctx);
}

#define DEFINE_UNARY_FUNC(name, T)                                         \
  template <>                                                              \
  DRAGON_API void name<T, MPSContext>(                                     \
      const int N, const float alpha, const T* x, T* y, MPSContext* ctx) { \
    auto kernel = MPSKernel::TypedString<T>(#name);                        \
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

DEFINE_UNARY_FUNC(Bias, uint8_t);
DEFINE_UNARY_FUNC(Bias, int8_t);
DEFINE_UNARY_FUNC(Bias, int);
DEFINE_UNARY_FUNC(Bias, int64_t);
DEFINE_UNARY_FUNC(Bias, float16);
DEFINE_UNARY_FUNC(Bias, float);
DEFINE_UNARY_FUNC(Bias, double);
DEFINE_UNARY_FUNC(InvStd, float16);
DEFINE_UNARY_FUNC(InvStd, float);
DEFINE_UNARY_FUNC(InvStd, double);
DEFINE_UNARY_FUNC(NaNToNum, float16);
DEFINE_UNARY_FUNC(NaNToNum, bfloat16);
DEFINE_UNARY_FUNC(NaNToNum, float);
DEFINE_UNARY_FUNC(NaNToNum, double);
#undef DEFINE_UNARY_FUNC

#define DEFINE_SET_FUNC(T)                                               \
  template <>                                                            \
  DRAGON_API void Set<T, MPSContext>(                                    \
      const int N, const T value, T* y, MPSContext* ctx) {               \
    if (value == T(0)) return ctx->MemsetAsync(sizeof(T) * N, y, 0);     \
    auto kernel = MPSKernel::TypedString<T>("Set");                      \
    const auto arg1 = float(value);                                      \
    vector<MPSConstant> args({MPSConstant(&arg1, MTLDataTypeFloat, 0)}); \
    auto* command_buffer = ctx->mps_stream()->command_buffer();          \
    auto* encoder = [command_buffer computeCommandEncoder];              \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);    \
    [encoder setComputePipelineState:pso];                               \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:0];             \
    MPSDispatchThreads(N, encoder, pso);                                 \
    [encoder endEncoding];                                               \
    [encoder release];                                                   \
  }

template <>
DRAGON_API void Set<float16, MPSContext>(
    const int N,
    const float16 value,
    float16* y,
    MPSContext* ctx) {
  if (value.x == (unsigned short)0) {
    return ctx->MemsetAsync(sizeof(float16) * N, y, 0);
  }
  const auto arg1 = convert::To<float>(value);
  vector<MPSConstant> args({MPSConstant(&arg1, MTLDataTypeFloat, 0)});
  auto* command_buffer = ctx->mps_stream()->command_buffer();
  auto* encoder = [command_buffer computeCommandEncoder];
  auto* pso = MPSKernel("Set_half", METAL_SHADERS).GetState(ctx, args);
  [encoder setComputePipelineState:pso];
  [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:0];
  MPSDispatchThreads(N, encoder, pso);
  [encoder endEncoding];
  [encoder release];
}

template <>
DRAGON_API void Set<bfloat16, MPSContext>(
    const int N,
    const bfloat16 value,
    bfloat16* y,
    MPSContext* ctx) {
  if (value.x == (unsigned short)0) {
    return ctx->MemsetAsync(sizeof(bfloat16) * N, y, 0);
  }
  const auto arg1 = convert::To<float>(value);
  vector<MPSConstant> args({MPSConstant(&arg1, MTLDataTypeFloat, 0)});
  auto* command_buffer = ctx->mps_stream()->command_buffer();
  auto* encoder = [command_buffer computeCommandEncoder];
  auto* pso = MPSKernel("Set_bfloat", METAL_SHADERS).GetState(ctx, args);
  [encoder setComputePipelineState:pso];
  [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:0];
  MPSDispatchThreads(N, encoder, pso);
  [encoder endEncoding];
  [encoder release];
}

DEFINE_SET_FUNC(bool);
DEFINE_SET_FUNC(uint8_t);
DEFINE_SET_FUNC(int8_t);
DEFINE_SET_FUNC(int);
DEFINE_SET_FUNC(int64_t);
DEFINE_SET_FUNC(float);
DEFINE_SET_FUNC(double);
#undef DEFINE_SET_FUNC

#define DEFINE_APPLY_MASK_FUNC(T)                                         \
  template <>                                                             \
  DRAGON_API void ApplyMask<T, MPSContext>(                               \
      const int N,                                                        \
      const float alpha,                                                  \
      const uint8_t* mask,                                                \
      const T* x,                                                         \
      T* y,                                                               \
      MPSContext* ctx) {                                                  \
    auto kernel = MPSKernel::TypedString<T>("ApplyMask");                 \
    vector<MPSConstant> args({MPSConstant(&alpha, MTLDataTypeFloat, 0)}); \
    auto* command_buffer = ctx->mps_stream()->command_buffer();           \
    auto* encoder = [command_buffer computeCommandEncoder];               \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);     \
    [encoder setComputePipelineState:pso];                                \
    [encoder setBuffer:id<MTLBuffer>(mask) offset:0 atIndex:0];           \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:1];              \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:2];              \
    MPSDispatchThreads(N, encoder, pso);                                  \
    [encoder endEncoding];                                                \
    [encoder release];                                                    \
  }

DEFINE_APPLY_MASK_FUNC(uint8_t);
DEFINE_APPLY_MASK_FUNC(int8_t);
DEFINE_APPLY_MASK_FUNC(int);
DEFINE_APPLY_MASK_FUNC(int64_t);
DEFINE_APPLY_MASK_FUNC(float16);
DEFINE_APPLY_MASK_FUNC(bfloat16);
DEFINE_APPLY_MASK_FUNC(float);
DEFINE_APPLY_MASK_FUNC(double);
#undef DEFINE_APPLY_MASK_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT)               \
  template <>                                                   \
  DRAGON_API void name<InputT, MPSContext>(                     \
      const int N,                                              \
      const InputT* a,                                          \
      const InputT* b,                                          \
      OutputT* y,                                               \
      MPSContext* ctx) {                                        \
    auto kernel = MPSKernel::TypedString<InputT>(#name);        \
    auto* command_buffer = ctx->mps_stream()->command_buffer(); \
    auto* encoder = [command_buffer computeCommandEncoder];     \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx); \
    [encoder setComputePipelineState:pso];                      \
    [encoder setBuffer:id<MTLBuffer>(a) offset:0 atIndex:0];    \
    [encoder setBuffer:id<MTLBuffer>(b) offset:0 atIndex:1];    \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:2];    \
    MPSDispatchThreads(N, encoder, pso);                        \
    [encoder endEncoding];                                      \
    [encoder release];                                          \
  }

DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Add, int8_t, int8_t);
DEFINE_BINARY_FUNC(Add, int, int);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t);
DEFINE_BINARY_FUNC(Add, float16, float16);
DEFINE_BINARY_FUNC(Add, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Add, float, float);
DEFINE_BINARY_FUNC(Add, double, double);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t);
DEFINE_BINARY_FUNC(Sub, int, int);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t);
DEFINE_BINARY_FUNC(Sub, float16, float16);
DEFINE_BINARY_FUNC(Sub, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Sub, float, float);
DEFINE_BINARY_FUNC(Sub, double, double);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t);
DEFINE_BINARY_FUNC(Mul, int, int);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t);
DEFINE_BINARY_FUNC(Mul, float16, float16);
DEFINE_BINARY_FUNC(Mul, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Mul, float, float);
DEFINE_BINARY_FUNC(Mul, double, double);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t);
DEFINE_BINARY_FUNC(Div, int, int);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t);
DEFINE_BINARY_FUNC(Div, float16, float16);
DEFINE_BINARY_FUNC(Div, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Div, float, float);
DEFINE_BINARY_FUNC(Div, double, double);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t);
DEFINE_BINARY_FUNC(Minimum, int, int);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t);
DEFINE_BINARY_FUNC(Minimum, float16, float16);
DEFINE_BINARY_FUNC(Minimum, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Minimum, float, float);
DEFINE_BINARY_FUNC(Minimum, double, double);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t);
DEFINE_BINARY_FUNC(Maximum, int, int);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t);
DEFINE_BINARY_FUNC(Maximum, float16, float16);
DEFINE_BINARY_FUNC(Maximum, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Maximum, float, float);
DEFINE_BINARY_FUNC(Maximum, double, double);
DEFINE_BINARY_FUNC(Pow, float16, float16);
DEFINE_BINARY_FUNC(Pow, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Pow, float, float);
DEFINE_BINARY_FUNC(Pow, double, double);
DEFINE_BINARY_FUNC(Atan2, float16, float16);
DEFINE_BINARY_FUNC(Atan2, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Atan2, float, float);
DEFINE_BINARY_FUNC(Atan2, double, double);
DEFINE_BINARY_FUNC(BitwiseAnd, bool, bool);
DEFINE_BINARY_FUNC(BitwiseAnd, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(BitwiseAnd, int8_t, int8_t);
DEFINE_BINARY_FUNC(BitwiseAnd, int, int);
DEFINE_BINARY_FUNC(BitwiseAnd, int64_t, int64_t);
DEFINE_BINARY_FUNC(BitwiseOr, bool, bool);
DEFINE_BINARY_FUNC(BitwiseOr, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(BitwiseOr, int8_t, int8_t);
DEFINE_BINARY_FUNC(BitwiseOr, int, int);
DEFINE_BINARY_FUNC(BitwiseOr, int64_t, int64_t);
DEFINE_BINARY_FUNC(BitwiseXor, bool, bool);
DEFINE_BINARY_FUNC(BitwiseXor, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(BitwiseXor, int8_t, int8_t);
DEFINE_BINARY_FUNC(BitwiseXor, int, int);
DEFINE_BINARY_FUNC(BitwiseXor, int64_t, int64_t);
DEFINE_BINARY_FUNC(And, bool, bool);
DEFINE_BINARY_FUNC(And, uint8_t, bool);
DEFINE_BINARY_FUNC(And, int8_t, bool);
DEFINE_BINARY_FUNC(And, int, bool);
DEFINE_BINARY_FUNC(And, int64_t, bool);
DEFINE_BINARY_FUNC(And, float16, bool);
DEFINE_BINARY_FUNC(And, bfloat16, bool);
DEFINE_BINARY_FUNC(And, float, bool);
DEFINE_BINARY_FUNC(And, double, bool);
DEFINE_BINARY_FUNC(Or, bool, bool);
DEFINE_BINARY_FUNC(Or, uint8_t, bool);
DEFINE_BINARY_FUNC(Or, int8_t, bool);
DEFINE_BINARY_FUNC(Or, int, bool);
DEFINE_BINARY_FUNC(Or, int64_t, bool);
DEFINE_BINARY_FUNC(Or, float16, bool);
DEFINE_BINARY_FUNC(Or, bfloat16, bool);
DEFINE_BINARY_FUNC(Or, float, bool);
DEFINE_BINARY_FUNC(Or, double, bool);
DEFINE_BINARY_FUNC(Xor, bool, bool);
DEFINE_BINARY_FUNC(Xor, uint8_t, bool);
DEFINE_BINARY_FUNC(Xor, int8_t, bool);
DEFINE_BINARY_FUNC(Xor, int, bool);
DEFINE_BINARY_FUNC(Xor, int64_t, bool);
DEFINE_BINARY_FUNC(Xor, float16, bool);
DEFINE_BINARY_FUNC(Xor, bfloat16, bool);
DEFINE_BINARY_FUNC(Xor, float, bool);
DEFINE_BINARY_FUNC(Xor, double, bool);
DEFINE_BINARY_FUNC(Equal, bool, bool);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool);
DEFINE_BINARY_FUNC(Equal, int8_t, bool);
DEFINE_BINARY_FUNC(Equal, int, bool);
DEFINE_BINARY_FUNC(Equal, int64_t, bool);
DEFINE_BINARY_FUNC(Equal, float16, bool);
DEFINE_BINARY_FUNC(Equal, bfloat16, bool);
DEFINE_BINARY_FUNC(Equal, float, bool);
DEFINE_BINARY_FUNC(Equal, double, bool);
DEFINE_BINARY_FUNC(NotEqual, bool, bool);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool);
DEFINE_BINARY_FUNC(NotEqual, int, bool);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool);
DEFINE_BINARY_FUNC(NotEqual, float16, bool);
DEFINE_BINARY_FUNC(NotEqual, bfloat16, bool);
DEFINE_BINARY_FUNC(NotEqual, float, bool);
DEFINE_BINARY_FUNC(NotEqual, double, bool);
DEFINE_BINARY_FUNC(Less, bool, bool);
DEFINE_BINARY_FUNC(Less, uint8_t, bool);
DEFINE_BINARY_FUNC(Less, int8_t, bool);
DEFINE_BINARY_FUNC(Less, int, bool);
DEFINE_BINARY_FUNC(Less, int64_t, bool);
DEFINE_BINARY_FUNC(Less, float16, bool);
DEFINE_BINARY_FUNC(Less, bfloat16, bool);
DEFINE_BINARY_FUNC(Less, float, bool);
DEFINE_BINARY_FUNC(Less, double, bool);
DEFINE_BINARY_FUNC(LessEqual, bool, bool);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool);
DEFINE_BINARY_FUNC(LessEqual, int, bool);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool);
DEFINE_BINARY_FUNC(LessEqual, float16, bool);
DEFINE_BINARY_FUNC(LessEqual, bfloat16, bool);
DEFINE_BINARY_FUNC(LessEqual, float, bool);
DEFINE_BINARY_FUNC(LessEqual, double, bool);
DEFINE_BINARY_FUNC(Greater, bool, bool);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool);
DEFINE_BINARY_FUNC(Greater, int8_t, bool);
DEFINE_BINARY_FUNC(Greater, int, bool);
DEFINE_BINARY_FUNC(Greater, int64_t, bool);
DEFINE_BINARY_FUNC(Greater, float16, bool);
DEFINE_BINARY_FUNC(Greater, bfloat16, bool);
DEFINE_BINARY_FUNC(Greater, float, bool);
DEFINE_BINARY_FUNC(Greater, double, bool);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, float16, bool);
DEFINE_BINARY_FUNC(GreaterEqual, bfloat16, bool);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool);
#undef DEFINE_BINARY_FUNC

} // namespace math

} // namespace dragon
