#include "dragon/utils/math/broadcast.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

template <typename T, int L, int N>
struct SimpleArray { vec<T, L> data[N]; };

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)

constant bool bool_arg1 [[function_constant(0)]];
constant uint uint_arg1 [[function_constant(1)]];
constant uint4 uint4_arg1 [[function_constant(2)]];
constant uint4 uint4_arg2 [[function_constant(3)]];
constant uint4 uint4_arg3 [[function_constant(4)]];
constant uint4 uint4_arg4 [[function_constant(5)]];
constant uint4 uint4_arg5 [[function_constant(6)]];
constant uint4 uint4_arg6 [[function_constant(7)]];
constant SimpleArray<uint, 4, 2> uintarr_arg1 = {uint4_arg1, uint4_arg2};
constant SimpleArray<uint, 4, 2> uintarr_arg2 = {uint4_arg3, uint4_arg4};
constant SimpleArray<uint, 4, 2> uintarr_arg3 = {uint4_arg5, uint4_arg6};

template <typename T>
kernel void RowwiseSet(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = x[index % uint_arg1];
}

template <typename T>
kernel void ColwiseSet(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = x[index / uint_arg1];
}

template <typename T>
kernel void BroadcastSet(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  uint xi = 0, tmp = index, r;
  for (int d = uint_arg1 - 1; d >= 0; --d) {
    const int d1 = d / 4, d2 = d % 4;
    FIXED_DIVISOR_DIV_MOD(uintarr_arg2.data[d1][d2], tmp, &tmp, &r);
    xi += r * uintarr_arg1.data[d1][d2];
  }
  y[index] = x[xi];
}

#define DEFINE_ROWWISE_BINARY_KERNEL(name, expr)                       \
  template <typename InputT, typename OutputT>                         \
  kernel void name(                                                    \
      device const InputT* a,                                          \
      device const InputT* b,                                          \
      device OutputT* y,                                               \
      const uint index [[thread_position_in_grid]]) {                  \
    const uint i = index % uint_arg1;                                  \
    y[index] = a[bool_arg1 ? i : index] expr b[bool_arg1 ? index : i]; \
  }

template <typename InputT, typename OutputT>
kernel void RowwiseXor(
    device const InputT* a,
    device const InputT* b,
    device OutputT* y,
    const uint index [[thread_position_in_grid]]) {
  const uint i = index % uint_arg1;
  y[index] = bool(a[bool_arg1 ? i : index]) ^ bool(b[bool_arg1 ? index : i]);
}

DEFINE_ROWWISE_BINARY_KERNEL(RowwiseAdd, +);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseSub, -);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseMul, *);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseDiv, /);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseBitwiseAnd, &);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseBitwiseOr, |);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseBitwiseXor, ^);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseAnd, &&);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseOr, ||);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseEqual, ==);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseNotEqual, !=);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseLess, <);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseLessEqual, <=);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseGreater, >);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseGreaterEqual, >=);
#undef DEFINE_ROWWISE_BINARY_KERNEL

#define DEFINE_ROWWISE_BINARY_KERNEL(name, func)                         \
  template <typename InputT, typename OutputT>                           \
  kernel void name(                                                      \
      device const InputT* a,                                            \
      device const InputT* b,                                            \
      device OutputT* y,                                                 \
      const uint index [[thread_position_in_grid]]) {                    \
    const uint i = index % uint_arg1;                                    \
    y[index] = func(a[bool_arg1 ? i : index], b[bool_arg1 ? index : i]); \
  }

DEFINE_ROWWISE_BINARY_KERNEL(RowwisePow, pow);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseAtan2, atan2);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseMinimum, min);
DEFINE_ROWWISE_BINARY_KERNEL(RowwiseMaximum, max);
#undef DEFINE_ROWWISE_BINARY_KERNEL

#define DEFINE_COLWISE_BINARY_KERNEL(name, expr)                       \
  template <typename InputT, typename OutputT>                         \
  kernel void name(                                                    \
      device const InputT* a,                                          \
      device const InputT* b,                                          \
      device OutputT* y,                                               \
      const uint index [[thread_position_in_grid]]) {                  \
    const uint i = index / uint_arg1;                                  \
    y[index] = a[bool_arg1 ? i : index] expr b[bool_arg1 ? index : i]; \
  }

template <typename InputT, typename OutputT>
kernel void ColwiseXor(
    device const InputT* a,
    device const InputT* b,
    device OutputT* y,
    const uint index [[thread_position_in_grid]]) {
  const uint i = index / uint_arg1;
  y[index] = bool(a[bool_arg1 ? i : index]) ^ bool(b[bool_arg1 ? index : i]);
}

DEFINE_COLWISE_BINARY_KERNEL(ColwiseAdd, +);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseSub, -);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseMul, *);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseDiv, /);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseBitwiseAnd, &);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseBitwiseOr, |);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseBitwiseXor, ^);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseAnd, &&);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseOr, ||);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseEqual, ==);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseNotEqual, !=);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseLess, <);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseLessEqual, <=);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseGreater, >);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseGreaterEqual, >=);
#undef DEFINE_COLWISE_BINARY_KERNEL

#define DEFINE_COLWISE_BINARY_KERNEL(name, func)                         \
  template <typename InputT, typename OutputT>                           \
  kernel void name(                                                      \
      device const InputT* a,                                            \
      device const InputT* b,                                            \
      device OutputT* y,                                                 \
      const uint index [[thread_position_in_grid]]) {                    \
    const uint i = index / uint_arg1;                                    \
    y[index] = func(a[bool_arg1 ? i : index], b[bool_arg1 ? index : i]); \
  }

DEFINE_COLWISE_BINARY_KERNEL(ColwisePow, pow);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseAtan2, atan2);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseMinimum, min);
DEFINE_COLWISE_BINARY_KERNEL(ColwiseMaximum, max);
#undef DEFINE_COLWISE_BINARY_KERNEL

#define DEFINE_BROADCAST_BINARY_KERNEL(name, expr)                     \
  template <typename InputT, typename OutputT>                         \
  kernel void name(                                                    \
      device const InputT* a,                                          \
      device const InputT* b,                                          \
      device OutputT* y,                                               \
      const uint index [[thread_position_in_grid]]) {                  \
    uint ai = 0, bi = 0, tmp = index, r;                               \
    for (int d = uint_arg1 - 1; d >= 0; --d) {                         \
      const int d1 = d / 4, d2 = d % 4;                                \
      FIXED_DIVISOR_DIV_MOD(uintarr_arg3.data[d1][d2], tmp, &tmp, &r); \
      ai += r * uintarr_arg1.data[d1][d2];                             \
      bi += r * uintarr_arg2.data[d1][d2];                             \
    }                                                                  \
    y[index] = a[ai] expr b[bi];                                       \
  }

template <typename InputT, typename OutputT>
kernel void BroadcastXor(
    device const InputT* a,
    device const InputT* b,
    device OutputT* y,
    const uint index [[thread_position_in_grid]]) {
  uint ai = 0, bi = 0, tmp = index, r;
  for (int d = uint_arg1 - 1; d >= 0; --d) {
    const int d1 = d / 4, d2 = d % 4;
    FIXED_DIVISOR_DIV_MOD(uintarr_arg3.data[d1][d2], tmp, &tmp, &r);
    ai += r * uintarr_arg1.data[d1][d2];
    bi += r * uintarr_arg2.data[d1][d2];
  }
  y[index] = bool(a[ai]) ^ bool(b[bi]);
}

DEFINE_BROADCAST_BINARY_KERNEL(BroadcastAdd, +);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastSub, -);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastMul, *);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastDiv, /);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastBitwiseAnd, &);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastBitwiseOr, |);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastBitwiseXor, ^);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastAnd, &&);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastOr, ||);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastEqual, ==);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastNotEqual, !=);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastLess, <);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastLessEqual, <=);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastGreater, >);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastGreaterEqual, >=);
#undef DEFINE_BROADCAST_BINARY_KERNEL

#define DEFINE_BROADCAST_BINARY_KERNEL(name, func)                     \
  template <typename InputT, typename OutputT>                         \
  kernel void name(                                                    \
      device const InputT* a,                                          \
      device const InputT* b,                                          \
      device OutputT* y,                                               \
      const uint index [[thread_position_in_grid]]) {                  \
    uint ai = 0, bi = 0, tmp = index, r;                               \
    for (int d = uint_arg1 - 1; d >= 0; --d) {                         \
      const int d1 = d / 4, d2 = d % 4;                                \
      FIXED_DIVISOR_DIV_MOD(uintarr_arg3.data[d1][d2], tmp, &tmp, &r); \
      ai += r * uintarr_arg1.data[d1][d2];                             \
      bi += r * uintarr_arg2.data[d1][d2];                             \
    }                                                                  \
    y[index] = func(a[ai], b[bi]);                                     \
  }

DEFINE_BROADCAST_BINARY_KERNEL(BroadcastPow, pow);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastAtan2, atan2);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastMinimum, min);
DEFINE_BROADCAST_BINARY_KERNEL(BroadcastMaximum, max);
#undef DEFINE_BROADCAST_BINARY_KERNEL

#define INSTANTIATE_SET_KERNEL(T) \
  template [[host_name("RowwiseSet_"#T)]] \
  kernel void RowwiseSet(device const T*, device T*, uint); \
  template [[host_name("ColwiseSet_"#T)]] \
  kernel void ColwiseSet(device const T*, device T*, uint); \
  template [[host_name("BroadcastSet_"#T)]] \
  kernel void BroadcastSet(device const T*, device T*, uint);

INSTANTIATE_SET_KERNEL(bool);
INSTANTIATE_SET_KERNEL(uint8_t);
INSTANTIATE_SET_KERNEL(int8_t);
INSTANTIATE_SET_KERNEL(int);
INSTANTIATE_SET_KERNEL(int64_t);
INSTANTIATE_SET_KERNEL(half);
INSTANTIATE_SET_KERNEL(float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_SET_KERNEL(double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_SET_KERNEL

#define INSTANTIATE_BINARY_KERNEL(name, InputT, OutputT) \
  template [[host_name("Rowwise"#name"_"#InputT)]] \
  kernel void Rowwise##name(device const InputT*, device const InputT*, \
                            device OutputT*, uint); \
  template [[host_name("Colwise"#name"_"#InputT)]] \
  kernel void Colwise##name(device const InputT*, device const InputT*, \
                            device OutputT*, uint); \
  template [[host_name("Broadcast"#name"_"#InputT)]] \
  kernel void Broadcast##name(device const InputT*, device const InputT*, \
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

#define DEFINE_SET_FUNC(T)                                                   \
  template <>                                                                \
  DRAGON_API void Set<T, MPSContext>(                                        \
      const int x_ndim,                                                      \
      const int64_t* x_dims,                                                 \
      const int y_ndim,                                                      \
      const int64_t* y_dims,                                                 \
      const T* x,                                                            \
      T* y,                                                                  \
      MPSContext* ctx) {                                                     \
    int64_t rows, cols;                                                      \
    vec64_t X_dims(x_dims, x_dims + x_ndim);                                 \
    vec64_t Y_dims(y_dims, y_dims + y_ndim);                                 \
    vec64_t X_broadcast_dims, Y_broadcast_dims;                              \
    math::utils::ComputeBroadcastDims(                                       \
        X_dims, Y_dims, X_broadcast_dims, Y_broadcast_dims);                 \
    if (X_broadcast_dims == Y_broadcast_dims) {                              \
      const auto N = math::utils::Prod(x_ndim, x_dims);                      \
      Copy(N, x, y, ctx);                                                    \
      return;                                                                \
    }                                                                        \
    auto args = vector<MPSConstant>();                                       \
    auto* command_buffer = ctx->mps_stream()->command_buffer();              \
    auto* encoder = [command_buffer computeCommandEncoder];                  \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                 \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];                 \
    MTLComputePipelineState_t pso = nil;                                     \
    if (math::utils::IsRowwiseBroadcast(X_dims, Y_dims, &rows, &cols)) {     \
      auto kernel = MPSKernel::TypedString<T>("RowwiseSet");                 \
      const auto arg1 = uint(cols);                                          \
      args.emplace_back(MPSConstant(&arg1, MTLDataTypeUInt, 1));             \
      pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);            \
      [encoder setComputePipelineState:pso];                                 \
      MPSDispatchThreads((rows * cols), encoder, pso);                       \
    } else if (math::utils::IsColwiseBroadcast(                              \
                   X_dims, Y_dims, &rows, &cols)) {                          \
      auto kernel = MPSKernel::TypedString<T>("ColwiseSet");                 \
      const auto arg1 = uint(cols);                                          \
      args.emplace_back(MPSConstant(&arg1, MTLDataTypeUInt, 1));             \
      pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);            \
      [encoder setComputePipelineState:pso];                                 \
      MPSDispatchThreads((rows * cols), encoder, pso);                       \
    } else {                                                                 \
      auto kernel = MPSKernel::TypedString<T>("BroadcastSet");               \
      vec64_t X_broadcast_strides, _;                                        \
      MPS_TENSOR_DIMS_CHECK(int(Y_dims.size()));                             \
      math::utils::ComputeBroadcastStrides(                                  \
          X_dims, Y_dims, X_broadcast_strides, _, _);                        \
      const auto arg1 = uint(Y_dims.size());                                 \
      vector<uint32_t> arg2(MPS_TENSOR_MAX_DIMS, 0);                         \
      vector<uint32_t> arg3(MPS_TENSOR_MAX_DIMS, 0);                         \
      for (size_t i = 0; i < X_broadcast_strides.size(); ++i) {              \
        arg2[i] = X_broadcast_strides[i];                                    \
      }                                                                      \
      for (size_t i = 0; i < Y_dims.size(); ++i) {                           \
        arg3[i] = Y_dims[i];                                                 \
      }                                                                      \
      args.emplace_back(MPSConstant(&arg1, MTLDataTypeUInt, 1));             \
      args.emplace_back(MPSConstant(arg2.data(), MTLDataTypeUInt4, {2, 3})); \
      args.emplace_back(MPSConstant(arg3.data(), MTLDataTypeUInt4, {4, 5})); \
      pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);            \
      [encoder setComputePipelineState:pso];                                 \
      MPSDispatchThreads(math::utils::Prod(Y_dims), encoder, pso);           \
    }                                                                        \
    [encoder endEncoding];                                                   \
    [encoder release];                                                       \
  }

DEFINE_SET_FUNC(bool);
DEFINE_SET_FUNC(uint8_t);
DEFINE_SET_FUNC(int8_t);
DEFINE_SET_FUNC(int);
DEFINE_SET_FUNC(int64_t);
DEFINE_SET_FUNC(float16);
DEFINE_SET_FUNC(bfloat16);
DEFINE_SET_FUNC(float);
DEFINE_SET_FUNC(double);
#undef DEFINE_SET_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT)                            \
  template <>                                                                \
  DRAGON_API void name<InputT, MPSContext>(                                  \
      const int a_ndim,                                                      \
      const int64_t* a_dims,                                                 \
      const int b_ndim,                                                      \
      const int64_t* b_dims,                                                 \
      const InputT* a,                                                       \
      const InputT* b,                                                       \
      OutputT* y,                                                            \
      MPSContext* ctx) {                                                     \
    int64_t rows, cols, broadcast_1st;                                       \
    vec64_t A_dims(a_dims, a_dims + a_ndim);                                 \
    vec64_t B_dims(b_dims, b_dims + b_ndim);                                 \
    vec64_t A_broadcast_dims, B_broadcast_dims;                              \
    math::utils::ComputeBroadcastDims(                                       \
        A_dims, B_dims, A_broadcast_dims, B_broadcast_dims);                 \
    if (A_broadcast_dims == B_broadcast_dims) {                              \
      const auto N = math::utils::Prod(a_ndim, a_dims);                      \
      name(N, a, b, y, ctx);                                                 \
      return;                                                                \
    }                                                                        \
    auto args = vector<MPSConstant>();                                       \
    auto* command_buffer = ctx->mps_stream()->command_buffer();              \
    auto* encoder = [command_buffer computeCommandEncoder];                  \
    [encoder setBuffer:id<MTLBuffer>(a) offset:0 atIndex:0];                 \
    [encoder setBuffer:id<MTLBuffer>(b) offset:0 atIndex:1];                 \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:2];                 \
    MTLComputePipelineState_t pso = nil;                                     \
    if (math::utils::IsRowwiseBroadcast(                                     \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {                 \
      auto kernel = MPSKernel::TypedString<InputT>("Rowwise" #name);         \
      const auto arg1 = bool(broadcast_1st);                                 \
      const auto arg2 = uint(cols);                                          \
      args.emplace_back(MPSConstant(&arg1, MTLDataTypeBool, 0));             \
      args.emplace_back(MPSConstant(&arg2, MTLDataTypeUInt, 1));             \
      pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);            \
      [encoder setComputePipelineState:pso];                                 \
      MPSDispatchThreads((rows * cols), encoder, pso);                       \
    } else if (math::utils::IsColwiseBroadcast(                              \
                   A_dims, B_dims, &rows, &cols, &broadcast_1st)) {          \
      auto kernel = MPSKernel::TypedString<InputT>("Colwise" #name);         \
      const auto arg1 = bool(broadcast_1st);                                 \
      const auto arg2 = uint(cols);                                          \
      args.emplace_back(MPSConstant(&arg1, MTLDataTypeBool, 0));             \
      args.emplace_back(MPSConstant(&arg2, MTLDataTypeUInt, 1));             \
      pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);            \
      [encoder setComputePipelineState:pso];                                 \
      MPSDispatchThreads((rows * cols), encoder, pso);                       \
    } else {                                                                 \
      auto kernel = MPSKernel::TypedString<InputT>("Broadcast" #name);       \
      vec64_t A_broadcast_strides, B_broadcast_strides, Y_dims;              \
      math::utils::ComputeBroadcastStrides(                                  \
          A_dims, B_dims, A_broadcast_strides, B_broadcast_strides, Y_dims); \
      MPS_TENSOR_DIMS_CHECK(int(Y_dims.size()));                             \
      const auto arg1 = uint(Y_dims.size());                                 \
      vector<uint32_t> arg2(MPS_TENSOR_MAX_DIMS, 0);                         \
      vector<uint32_t> arg3(MPS_TENSOR_MAX_DIMS, 0);                         \
      vector<uint32_t> arg4(MPS_TENSOR_MAX_DIMS, 0);                         \
      for (size_t i = 0; i < A_broadcast_strides.size(); ++i) {              \
        arg2[i] = A_broadcast_strides[i];                                    \
      }                                                                      \
      for (size_t i = 0; i < B_broadcast_strides.size(); ++i) {              \
        arg3[i] = B_broadcast_strides[i];                                    \
      }                                                                      \
      for (size_t i = 0; i < Y_dims.size(); ++i) {                           \
        arg4[i] = Y_dims[i];                                                 \
      }                                                                      \
      args.emplace_back(MPSConstant(&arg1, MTLDataTypeUInt, 1));             \
      args.emplace_back(MPSConstant(arg2.data(), MTLDataTypeUInt4, {2, 3})); \
      args.emplace_back(MPSConstant(arg3.data(), MTLDataTypeUInt4, {4, 5})); \
      args.emplace_back(MPSConstant(arg4.data(), MTLDataTypeUInt4, {6, 7})); \
      pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);            \
      [encoder setComputePipelineState:pso];                                 \
      MPSDispatchThreads(math::utils::Prod(Y_dims), encoder, pso);           \
    };                                                                       \
    [encoder endEncoding];                                                   \
    [encoder release];                                                       \
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
