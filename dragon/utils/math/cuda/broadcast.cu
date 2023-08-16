#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/broadcast.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/types.h"

namespace dragon {

namespace math {

namespace {

template <typename T>
__global__ void _RowwiseSet(const int N, const int cols, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = math::utils::LDG(x + i % cols);
  }
}

template <typename T>
__global__ void _ColwiseSet(const int N, const int cols, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = math::utils::LDG(x + i / cols);
  }
}

template <typename T, int D>
__global__ void _BroadcastSet(
    const int N,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += r * x_strides.data[d];
    }
    y[yi] = math::utils::LDG(x + xi);
  }
}

template <typename InputT, typename OutputT, class Functor, bool BroadcastA>
__global__ void _RowwiseBinaryFunc(
    const int N,
    const int cols,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    const int i = yi % cols;
    const int ai = BroadcastA ? i : yi;
    const int bi = BroadcastA ? yi : i;
    y[yi] = op(math::utils::LDG(a + ai), math::utils::LDG(b + bi));
  }
}

template <typename InputT, typename OutputT, class Functor, bool BroadcastA>
__global__ void _ColwiseBinaryFunc(
    const int N,
    const int cols,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    const int i = yi / cols;
    const int ai = BroadcastA ? i : yi;
    const int bi = BroadcastA ? yi : i;
    y[yi] = op(math::utils::LDG(a + ai), math::utils::LDG(b + bi));
  }
}

template <typename InputT, typename OutputT, class Functor, int D>
__global__ void _BroadcastBinaryFunc(
    const int N,
    const SimpleArray<int, D> a_strides,
    const SimpleArray<int, D> b_strides,
    const SimpleArray<int, D> y_dims,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int ai = 0, bi = 0, tmp = yi;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      ai += r * a_strides.data[d];
      bi += r * b_strides.data[d];
    }
    y[yi] = op(math::utils::LDG(a + ai), math::utils::LDG(b + bi));
  }
}

template <typename T, int D>
__global__ void _BroadcastWhere(
    const int N,
    const SimpleArray<int, D> a_strides,
    const SimpleArray<int, D> b_strides,
    const SimpleArray<int, D> c_strides,
    const SimpleArray<int, D> y_dims,
    const T* a,
    const T* b,
    const uint8_t* c,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int ai = 0, bi = 0, ci = 0, tmp = yi;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      ai += r * a_strides.data[d];
      bi += r * b_strides.data[d];
      ci += r * c_strides.data[d];
    }
    y[yi] = math::utils::LDG(c + ci) ? math::utils::LDG(a + ai)
                                     : math::utils::LDG(b + bi);
  }
}

template <typename T, int D>
void _BroadcastSetImpl(
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  SimpleArray<int, D> X_strides, Y_dims;
  const auto N = math::utils::Prod(D, y_dims);
  for (int i = 0; i < D; ++i) {
    X_strides.data[i] = x_strides[i];
    Y_dims.data[i] = y_dims[i];
  }
  _BroadcastSet<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, X_strides, Y_dims, x, y);
}

template <typename InputT, typename OutputT, class Functor, int D>
void _BroadcastBinaryFuncImpl(
    const int64_t* a_strides,
    const int64_t* b_strides,
    const int64_t* y_dims,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y,
    CUDAContext* ctx) {
  SimpleArray<int, D> A_strides, B_strides, Y_dims;
  const auto N = math::utils::Prod(D, y_dims);
  for (int i = 0; i < D; ++i) {
    A_strides.data[i] = a_strides[i];
    B_strides.data[i] = b_strides[i];
    Y_dims.data[i] = y_dims[i];
  }
  _BroadcastBinaryFunc<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, A_strides, B_strides, Y_dims, op, a, b, y);
}

template <typename T, int D>
void _BroadcastWhereImpl(
    const int64_t* a_strides,
    const int64_t* b_strides,
    const int64_t* c_strides,
    const int64_t* y_dims,
    const T* a,
    const T* b,
    const uint8_t* c,
    T* y,
    CUDAContext* ctx) {
  SimpleArray<int, D> A_strides, B_strides, C_strides;
  SimpleArray<int, D> Y_dims;
  const auto N = math::utils::Prod(D, y_dims);
  for (int i = 0; i < D; ++i) {
    A_strides.data[i] = a_strides[i];
    B_strides.data[i] = b_strides[i];
    C_strides.data[i] = c_strides[i];
    Y_dims.data[i] = y_dims[i];
  }
  _BroadcastWhere<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, A_strides, B_strides, C_strides, Y_dims, a, b, c, y);
}

} // namespace

#define DEFINE_SET_FUNC(T, AliasT)                                          \
  template <>                                                               \
  DRAGON_API void Set<T, CUDAContext>(                                      \
      const int x_ndim,                                                     \
      const int64_t* x_dims,                                                \
      const int y_ndim,                                                     \
      const int64_t* y_dims,                                                \
      const T* x,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    int64_t rows, cols;                                                     \
    vec64_t X_dims(x_dims, x_dims + x_ndim);                                \
    vec64_t Y_dims(y_dims, y_dims + y_ndim);                                \
    vec64_t X_broadcast_dims, Y_broadcast_dims;                             \
    math::utils::ComputeBroadcastDims(                                      \
        X_dims, Y_dims, X_broadcast_dims, Y_broadcast_dims);                \
    if (X_broadcast_dims == Y_broadcast_dims) {                             \
      const auto N = math::utils::Prod(x_ndim, x_dims);                     \
      Copy(N, x, y, ctx);                                                   \
      return;                                                               \
    }                                                                       \
    if (math::utils::IsRowwiseBroadcast(X_dims, Y_dims, &rows, &cols)) {    \
      const auto N = rows * cols;                                           \
      _RowwiseSet<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          N,                                                                \
          cols,                                                             \
          reinterpret_cast<const AliasT*>(x),                               \
          reinterpret_cast<AliasT*>(y));                                    \
      return;                                                               \
    }                                                                       \
    if (math::utils::IsColwiseBroadcast(X_dims, Y_dims, &rows, &cols)) {    \
      const auto N = rows * cols;                                           \
      _ColwiseSet<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          N,                                                                \
          cols,                                                             \
          reinterpret_cast<const AliasT*>(x),                               \
          reinterpret_cast<AliasT*>(y));                                    \
      return;                                                               \
    }                                                                       \
    vec64_t X_broadcast_strides, _;                                         \
    CUDA_TENSOR_DIMS_CHECK(int(Y_dims.size()));                             \
    math::utils::ComputeBroadcastStrides(                                   \
        X_dims, Y_dims, X_broadcast_strides, _, _);                         \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1(                                     \
        _BroadcastSetImpl,                                                  \
        AliasT,                                                             \
        int(Y_dims.size()),                                                 \
        X_broadcast_strides.data(),                                         \
        Y_dims.data(),                                                      \
        reinterpret_cast<const AliasT*>(x),                                 \
        reinterpret_cast<AliasT*>(y),                                       \
        ctx);                                                               \
  }

DEFINE_SET_FUNC(bool, uint8_t);
DEFINE_SET_FUNC(uint8_t, uint8_t);
DEFINE_SET_FUNC(int8_t, int8_t);
DEFINE_SET_FUNC(int, int);
DEFINE_SET_FUNC(int64_t, int64_t);
DEFINE_SET_FUNC(float16, half);
DEFINE_SET_FUNC(bfloat16, nv_bfloat16);
DEFINE_SET_FUNC(float, float);
DEFINE_SET_FUNC(double, double);
#undef DEFINE_SET_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, Functor)                   \
  template <>                                                                \
  DRAGON_API void name<InputT, CUDAContext>(                                 \
      const int a_ndim,                                                      \
      const int64_t* a_dims,                                                 \
      const int b_ndim,                                                      \
      const int64_t* b_dims,                                                 \
      const InputT* a,                                                       \
      const InputT* b,                                                       \
      OutputT* y,                                                            \
      CUDAContext* ctx) {                                                    \
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
    if (math::utils::IsRowwiseBroadcast(                                     \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {                 \
      const auto N = rows * cols;                                            \
      if (broadcast_1st > 0) {                                               \
        _RowwiseBinaryFunc<                                                  \
            math::Traits<InputT>::scalar_type,                               \
            math::Traits<OutputT>::scalar_type,                              \
            Functor<math::Traits<InputT>::scalar_type>,                      \
            true><<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
            N,                                                               \
            cols,                                                            \
            Functor<math::Traits<InputT>::scalar_type>(),                    \
            reinterpret_cast<const math::Traits<InputT>::scalar_type*>(a),   \
            reinterpret_cast<const math::Traits<InputT>::scalar_type*>(b),   \
            reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y));       \
      } else {                                                               \
        _RowwiseBinaryFunc<                                                  \
            math::Traits<InputT>::scalar_type,                               \
            math::Traits<OutputT>::scalar_type,                              \
            Functor<math::Traits<InputT>::scalar_type>,                      \
            false><<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
            N,                                                               \
            cols,                                                            \
            Functor<math::Traits<InputT>::scalar_type>(),                    \
            reinterpret_cast<const math::Traits<InputT>::scalar_type*>(a),   \
            reinterpret_cast<const math::Traits<InputT>::scalar_type*>(b),   \
            reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y));       \
      }                                                                      \
      return;                                                                \
    }                                                                        \
    if (math::utils::IsColwiseBroadcast(                                     \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {                 \
      const auto N = rows * cols;                                            \
      if (broadcast_1st > 0) {                                               \
        _ColwiseBinaryFunc<                                                  \
            math::Traits<InputT>::scalar_type,                               \
            math::Traits<OutputT>::scalar_type,                              \
            Functor<math::Traits<InputT>::scalar_type>,                      \
            true><<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
            N,                                                               \
            cols,                                                            \
            Functor<math::Traits<InputT>::scalar_type>(),                    \
            reinterpret_cast<const math::Traits<InputT>::scalar_type*>(a),   \
            reinterpret_cast<const math::Traits<InputT>::scalar_type*>(b),   \
            reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y));       \
      } else {                                                               \
        _ColwiseBinaryFunc<                                                  \
            math::Traits<InputT>::scalar_type,                               \
            math::Traits<OutputT>::scalar_type,                              \
            Functor<math::Traits<InputT>::scalar_type>,                      \
            false><<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
            N,                                                               \
            cols,                                                            \
            Functor<math::Traits<InputT>::scalar_type>(),                    \
            reinterpret_cast<const math::Traits<InputT>::scalar_type*>(a),   \
            reinterpret_cast<const math::Traits<InputT>::scalar_type*>(b),   \
            reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y));       \
      }                                                                      \
      return;                                                                \
    }                                                                        \
    vec64_t A_broadcast_strides, B_broadcast_strides, Y_dims;                \
    math::utils::ComputeBroadcastStrides(                                    \
        A_dims, B_dims, A_broadcast_strides, B_broadcast_strides, Y_dims);   \
    CUDA_TENSOR_DIMS_CHECK(int(Y_dims.size()));                              \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_3(                                      \
        _BroadcastBinaryFuncImpl,                                            \
        math::Traits<InputT>::scalar_type,                                   \
        math::Traits<OutputT>::scalar_type,                                  \
        Functor<math::Traits<InputT>::scalar_type>,                          \
        int(Y_dims.size()),                                                  \
        A_broadcast_strides.data(),                                          \
        B_broadcast_strides.data(),                                          \
        Y_dims.data(),                                                       \
        Functor<math::Traits<InputT>::scalar_type>(),                        \
        reinterpret_cast<const math::Traits<InputT>::scalar_type*>(a),       \
        reinterpret_cast<const math::Traits<InputT>::scalar_type*>(b),       \
        reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y),            \
        ctx);                                                                \
  }

DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int8_t, int8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int, int, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, float16, float16, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, bfloat16, bfloat16, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, float, float, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, double, double, math::PlusFunctor);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int, int, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, float16, float16, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, bfloat16, bfloat16, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, float, float, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, double, double, math::MinusFunctor);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int, int, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, float16, float16, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, bfloat16, bfloat16, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, float, float, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, double, double, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int, int, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, float16, float16, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, bfloat16, bfloat16, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, float, float, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, double, double, math::DividesFunctor);
DEFINE_BINARY_FUNC(Pow, float16, float16, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, bfloat16, bfloat16, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, float, float, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, double, double, math::PowFunctor);
DEFINE_BINARY_FUNC(Atan2, float16, float16, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, bfloat16, bfloat16, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, float, float, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, double, double, math::Atan2Functor);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int, int, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, float16, float16, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, bfloat16, bfloat16, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, float, float, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, double, double, math::MinFunctor);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int, int, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, float16, float16, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, bfloat16, bfloat16, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, float, float, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, double, double, math::MaxFunctor);
DEFINE_BINARY_FUNC(BitwiseAnd, uint8_t, uint8_t, math::BitAndFunctor);
DEFINE_BINARY_FUNC(BitwiseAnd, int8_t, int8_t, math::BitAndFunctor);
DEFINE_BINARY_FUNC(BitwiseAnd, int, int, math::BitAndFunctor);
DEFINE_BINARY_FUNC(BitwiseAnd, int64_t, int64_t, math::BitAndFunctor);
DEFINE_BINARY_FUNC(BitwiseOr, uint8_t, uint8_t, math::BitOrFunctor);
DEFINE_BINARY_FUNC(BitwiseOr, int8_t, int8_t, math::BitOrFunctor);
DEFINE_BINARY_FUNC(BitwiseOr, int, int, math::BitOrFunctor);
DEFINE_BINARY_FUNC(BitwiseOr, int64_t, int64_t, math::BitOrFunctor);
DEFINE_BINARY_FUNC(BitwiseXor, uint8_t, uint8_t, math::BitXorFunctor);
DEFINE_BINARY_FUNC(BitwiseXor, int8_t, int8_t, math::BitXorFunctor);
DEFINE_BINARY_FUNC(BitwiseXor, int, int, math::BitXorFunctor);
DEFINE_BINARY_FUNC(BitwiseXor, int64_t, int64_t, math::BitXorFunctor);
DEFINE_BINARY_FUNC(And, uint8_t, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, int8_t, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, int, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, int64_t, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, float16, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, bfloat16, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, float, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, double, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(Or, uint8_t, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, int8_t, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, int, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, int64_t, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, float16, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, bfloat16, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, float, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, double, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Xor, uint8_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int8_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int64_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, float16, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, bfloat16, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, float, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, double, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, float16, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, bfloat16, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, float, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, double, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, float16, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, bfloat16, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, float, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, double, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int64_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, float16, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, bfloat16, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, float, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, double, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, float16, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, bfloat16, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, float, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, double, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, float16, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, bfloat16, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, float, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, double, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, float16, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, bfloat16, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool, math::GreaterEqualFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, InputAliasT, OutputAliasT) \
  template <>                                                                \
  DRAGON_API void name<InputT, CUDAContext>(                                 \
      const int a_ndim,                                                      \
      const int64_t* a_dims,                                                 \
      const int b_ndim,                                                      \
      const int64_t* b_dims,                                                 \
      const InputT* a,                                                       \
      const InputT* b,                                                       \
      OutputT* y,                                                            \
      CUDAContext* ctx) {                                                    \
    name(                                                                    \
        a_ndim,                                                              \
        a_dims,                                                              \
        b_ndim,                                                              \
        b_dims,                                                              \
        reinterpret_cast<const InputAliasT*>(a),                             \
        reinterpret_cast<const InputAliasT*>(b),                             \
        reinterpret_cast<OutputAliasT*>(y),                                  \
        ctx);                                                                \
  }

DEFINE_BINARY_FUNC(BitwiseAnd, bool, bool, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(BitwiseOr, bool, bool, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(BitwiseXor, bool, bool, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(And, bool, bool, uint8_t, bool);
DEFINE_BINARY_FUNC(Or, bool, bool, uint8_t, bool);
DEFINE_BINARY_FUNC(Xor, bool, bool, uint8_t, bool);
DEFINE_BINARY_FUNC(Equal, bool, bool, uint8_t, bool);
DEFINE_BINARY_FUNC(NotEqual, bool, bool, uint8_t, bool);
DEFINE_BINARY_FUNC(Less, bool, bool, uint8_t, bool);
DEFINE_BINARY_FUNC(LessEqual, bool, bool, uint8_t, bool);
DEFINE_BINARY_FUNC(Greater, bool, bool, uint8_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool, uint8_t, bool);
#undef DEFINE_BINARY_FUNC

#define DEFINE_WHERE_FUNC(T, AliasT)                                        \
  template <>                                                               \
  DRAGON_API void Where<T, CUDAContext>(                                    \
      const int a_ndim,                                                     \
      const int64_t* a_dims,                                                \
      const int b_ndim,                                                     \
      const int64_t* b_dims,                                                \
      const int c_ndim,                                                     \
      const int64_t* c_dims,                                                \
      const T* a,                                                           \
      const T* b,                                                           \
      const bool* c,                                                        \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    vec64_t A_dims(a_dims, a_dims + a_ndim);                                \
    vec64_t B_dims(b_dims, b_dims + b_ndim);                                \
    vec64_t C_dims(c_dims, c_dims + c_ndim);                                \
    vec64_t A_broadcast_dims, B_broadcast_dims, C_broadcast_dims;           \
    vec64_t A_broadcast_strides, B_broadcast_strides, C_broadcast_strides;  \
    vec64_t Y_dims, _, __;                                                  \
    math::utils::ComputeBroadcastStrides(A_dims, B_dims, _, _, __);         \
    math::utils::ComputeBroadcastStrides(C_dims, __, _, _, Y_dims);         \
    math::utils::ComputeBroadcastDims(A_dims, Y_dims, A_broadcast_dims, _); \
    math::utils::ComputeBroadcastDims(B_dims, Y_dims, B_broadcast_dims, _); \
    math::utils::ComputeBroadcastDims(C_dims, Y_dims, C_broadcast_dims, _); \
    if (A_broadcast_dims == B_broadcast_dims &&                             \
        B_broadcast_dims == C_broadcast_dims) {                             \
      const auto N = math::utils::Prod(a_ndim, a_dims);                     \
      Where(N, a, b, c, y, ctx);                                            \
      return;                                                               \
    }                                                                       \
    CUDA_TENSOR_DIMS_CHECK((int)Y_dims.size());                             \
    math::utils::ComputeBroadcastStrides(                                   \
        A_dims, Y_dims, A_broadcast_strides, _, _);                         \
    math::utils::ComputeBroadcastStrides(                                   \
        B_dims, Y_dims, B_broadcast_strides, _, _);                         \
    math::utils::ComputeBroadcastStrides(                                   \
        C_dims, Y_dims, C_broadcast_strides, _, _);                         \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1(                                     \
        _BroadcastWhereImpl,                                                \
        AliasT,                                                             \
        int(Y_dims.size()),                                                 \
        A_broadcast_strides.data(),                                         \
        B_broadcast_strides.data(),                                         \
        C_broadcast_strides.data(),                                         \
        Y_dims.data(),                                                      \
        reinterpret_cast<const AliasT*>(a),                                 \
        reinterpret_cast<const AliasT*>(b),                                 \
        reinterpret_cast<const uint8_t*>(c),                                \
        reinterpret_cast<AliasT*>(y),                                       \
        ctx);                                                               \
  }

DEFINE_WHERE_FUNC(bool, uint8_t);
DEFINE_WHERE_FUNC(uint8_t, uint8_t);
DEFINE_WHERE_FUNC(int8_t, int8_t);
DEFINE_WHERE_FUNC(int, int);
DEFINE_WHERE_FUNC(int64_t, int64_t);
DEFINE_WHERE_FUNC(float16, half);
DEFINE_WHERE_FUNC(bfloat16, nv_bfloat16);
DEFINE_WHERE_FUNC(float, float);
DEFINE_WHERE_FUNC(double, double);
#undef DEFINE_WHERE_FUNC

} // namespace math

} // namespace dragon
