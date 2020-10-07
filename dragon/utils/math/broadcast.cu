#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/broadcast.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

template <typename T>
__global__ void _RowwiseSet(const int n, const int cols, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i % cols);
#else
    y[i] = x[i % cols];
#endif
  }
}

template <typename T>
__global__ void _ColwiseSet(const int n, const int cols, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i / cols);
#else
    y[i] = x[i / cols];
#endif
  }
}

template <typename T, int D>
__global__ void _BroadcastSet(
    const int nthreads,
    const int num_dims,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += r * x_strides.data[d];
    }
#if __CUDA_ARCH__ >= 350
    y[yi] = __ldg(x + xi);
#else
    y[yi] = x[xi];
#endif
  }
}

template <typename TIn, typename TOut, class Operator, bool BroadcastA>
__global__ void _RowwiseBinaryFunc(
    const int nthreads,
    const int cols,
    const Operator op,
    const TIn* a,
    const TIn* b,
    TOut* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi % cols;
    const int ai = BroadcastA ? i : yi;
    const int bi = BroadcastA ? yi : i;
#if __CUDA_ARCH__ >= 350
    y[yi] = op(__ldg(a + ai), __ldg(b + bi));
#else
    y[yi] = op(a[ai], b[bi]);
#endif
  }
}

template <typename TIn, typename TOut, class Operator, bool BroadcastA>
__global__ void _ColwiseBinaryFunc(
    const int nthreads,
    const int cols,
    const Operator op,
    const TIn* a,
    const TIn* b,
    TOut* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi / cols;
    const int ai = BroadcastA ? i : yi;
    const int bi = BroadcastA ? yi : i;
#if __CUDA_ARCH__ >= 350
    y[yi] = op(__ldg(a + ai), __ldg(b + bi));
#else
    y[yi] = op(a[ai], b[bi]);
#endif
  }
}

template <typename TIn, typename TOut, class Operator, int D>
__global__ void _BroadcastBinaryFunc(
    const int nthreads,
    const int num_dims,
    const SimpleArray<int, D> a_strides,
    const SimpleArray<int, D> b_strides,
    const SimpleArray<int, D> y_dims,
    const Operator op,
    const TIn* a,
    const TIn* b,
    TOut* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int ai = 0, bi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      ai += r * a_strides.data[d];
      bi += r * b_strides.data[d];
    }
#if __CUDA_ARCH__ >= 350
    y[yi] = op(__ldg(a + ai), __ldg(b + bi));
#else
    y[yi] = op(a[ai], b[bi]);
#endif
  }
}

template <typename T, int D>
__global__ void _BroadcastWhere(
    const int nthreads,
    const int num_dims,
    const SimpleArray<int, D> a_strides,
    const SimpleArray<int, D> b_strides,
    const SimpleArray<int, D> c_strides,
    const SimpleArray<int, D> y_dims,
    const T* a,
    const T* b,
    const uint8_t* c,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int ai = 0, bi = 0, ci = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      ai += r * a_strides.data[d];
      bi += r * b_strides.data[d];
      ci += r * c_strides.data[d];
    }
#if __CUDA_ARCH__ >= 350
    y[yi] = __ldg(c + ci) ? __ldg(a + ai) : __ldg(b + bi);
#else
    y[yi] = c[ci] ? a[ai] : b[bi];
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_SET_FUNC(T1, T2)                                          \
  template <>                                                            \
  DRAGON_API void Set<T1, CUDAContext>(                                  \
      const int x_ndim,                                                  \
      const int64_t* x_dims,                                             \
      const int y_ndim,                                                  \
      const int64_t* y_dims,                                             \
      const T1* x,                                                       \
      T1* y,                                                             \
      CUDAContext* ctx) {                                                \
    int rows, cols;                                                      \
    vec64_t X_dims(x_dims, x_dims + x_ndim);                             \
    vec64_t Y_dims(y_dims, y_dims + y_ndim);                             \
    vec64_t X_broadcast_dims, Y_broadcast_dims;                          \
    utils::math::ComputeBinaryBroadcastDims(                             \
        X_dims, Y_dims, X_broadcast_dims, Y_broadcast_dims);             \
    if (X_broadcast_dims == Y_broadcast_dims) {                          \
      auto count = std::accumulate(                                      \
          x_dims, x_dims + x_ndim, 1, std::multiplies<int64_t>());       \
      Copy(count, x, y, ctx);                                            \
      return;                                                            \
    }                                                                    \
    if (utils::math::IsRowwiseBroadcast(X_dims, Y_dims, &rows, &cols)) { \
      const auto nthreads = rows * cols;                                 \
      _RowwiseSet<<<                                                     \
          CUDA_BLOCKS(nthreads),                                         \
          CUDA_THREADS,                                                  \
          0,                                                             \
          ctx->cuda_stream()>>>(                                         \
          nthreads,                                                      \
          cols,                                                          \
          reinterpret_cast<const T2*>(x),                                \
          reinterpret_cast<T2*>(y));                                     \
      return;                                                            \
    }                                                                    \
    if (utils::math::IsColwiseBroadcast(X_dims, Y_dims, &rows, &cols)) { \
      const auto nthreads = rows * cols;                                 \
      _ColwiseSet<<<                                                     \
          CUDA_BLOCKS(nthreads),                                         \
          CUDA_THREADS,                                                  \
          0,                                                             \
          ctx->cuda_stream()>>>(                                         \
          nthreads,                                                      \
          cols,                                                          \
          reinterpret_cast<const T2*>(x),                                \
          reinterpret_cast<T2*>(y));                                     \
      return;                                                            \
    }                                                                    \
    vec64_t X_broadcast_strides, _;                                      \
    CUDA_TENSOR_DIMS_CHECK((int)Y_dims.size());                          \
    utils::math::ComputeBinaryBroadcastStrides(                          \
        X_dims, Y_dims, X_broadcast_strides, _, _);                      \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> strides, dims;                \
    const auto nthreads = std::accumulate(                               \
        Y_dims.begin(), Y_dims.end(), 1, std::multiplies<int64_t>());    \
    for (int i = 0; i < Y_dims.size(); ++i) {                            \
      strides.data[i] = X_broadcast_strides[i];                          \
      dims.data[i] = Y_dims[i];                                          \
    }                                                                    \
    _BroadcastSet<<<                                                     \
        CUDA_BLOCKS(nthreads),                                           \
        CUDA_THREADS,                                                    \
        0,                                                               \
        ctx->cuda_stream()>>>(                                           \
        nthreads,                                                        \
        Y_dims.size(),                                                   \
        strides,                                                         \
        dims,                                                            \
        reinterpret_cast<const T2*>(x),                                  \
        reinterpret_cast<T2*>(y));                                       \
  }

DEFINE_SET_FUNC(bool, uint8_t);
DEFINE_SET_FUNC(int8_t, int8_t);
DEFINE_SET_FUNC(uint8_t, uint8_t);
DEFINE_SET_FUNC(int, int);
DEFINE_SET_FUNC(int64_t, int64_t);
DEFINE_SET_FUNC(float, float);
DEFINE_SET_FUNC(float16, half);
DEFINE_SET_FUNC(double, double);
#undef DEFINE_SET_FUNC

#define DEFINE_BINARY_FUNC(name, TIn, TOut, Functor)                          \
  template <>                                                                 \
  DRAGON_API void name<TIn, CUDAContext>(                                     \
      const int a_ndim,                                                       \
      const int64_t* a_dims,                                                  \
      const int b_ndim,                                                       \
      const int64_t* b_dims,                                                  \
      const TIn* a,                                                           \
      const TIn* b,                                                           \
      TOut* y,                                                                \
      CUDAContext* ctx) {                                                     \
    int rows, cols, broadcast_1st;                                            \
    vec64_t A_dims(a_dims, a_dims + a_ndim);                                  \
    vec64_t B_dims(b_dims, b_dims + b_ndim);                                  \
    vec64_t A_broadcast_dims, B_broadcast_dims;                               \
    utils::math::ComputeBinaryBroadcastDims(                                  \
        A_dims, B_dims, A_broadcast_dims, B_broadcast_dims);                  \
    if (A_broadcast_dims == B_broadcast_dims) {                               \
      auto count = std::accumulate(                                           \
          a_dims, a_dims + a_ndim, 1, std::multiplies<int64_t>());            \
      name(count, a, b, y, ctx);                                              \
      return;                                                                 \
    }                                                                         \
    if (utils::math::IsRowwiseBroadcast(                                      \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {                  \
      const auto nthreads = rows * cols;                                      \
      if (broadcast_1st > 0) {                                                \
        _RowwiseBinaryFunc<TIn, TOut, Functor<TIn>, true>                     \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads, cols, Functor<TIn>(), a, b, y);                     \
      } else {                                                                \
        _RowwiseBinaryFunc<TIn, TOut, Functor<TIn>, false>                    \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads, cols, Functor<TIn>(), a, b, y);                     \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
    if (utils::math::IsColwiseBroadcast(                                      \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {                  \
      const auto nthreads = rows * cols;                                      \
      if (broadcast_1st > 0) {                                                \
        _ColwiseBinaryFunc<TIn, TOut, Functor<TIn>, true>                     \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads, cols, Functor<TIn>(), a, b, y);                     \
      } else {                                                                \
        _ColwiseBinaryFunc<TIn, TOut, Functor<TIn>, false>                    \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads, cols, Functor<TIn>(), a, b, y);                     \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
    vec64_t A_broadcast_strides, B_broadcast_strides, Y_dims;                 \
    utils::math::ComputeBinaryBroadcastStrides(                               \
        A_dims, B_dims, A_broadcast_strides, B_broadcast_strides, Y_dims);    \
    CUDA_TENSOR_DIMS_CHECK((int)Y_dims.size());                               \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> a_strides, b_strides, y_dims;      \
    const auto nthreads = std::accumulate(                                    \
        Y_dims.begin(), Y_dims.end(), 1, std::multiplies<int64_t>());         \
    for (int i = 0; i < Y_dims.size(); ++i) {                                 \
      a_strides.data[i] = A_broadcast_strides[i];                             \
      b_strides.data[i] = B_broadcast_strides[i];                             \
      y_dims.data[i] = Y_dims[i];                                             \
    }                                                                         \
    _BroadcastBinaryFunc<TIn, TOut, Functor<TIn>, CUDA_TENSOR_MAX_DIMS>       \
        <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
            nthreads,                                                         \
            Y_dims.size(),                                                    \
            a_strides,                                                        \
            b_strides,                                                        \
            y_dims,                                                           \
            Functor<TIn>(),                                                   \
            a,                                                                \
            b,                                                                \
            y);                                                               \
  }

DEFINE_BINARY_FUNC(Add, int8_t, int8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int, int, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, float, float, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, double, double, math::PlusFunctor);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int, int, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, float, float, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, double, double, math::MinusFunctor);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int, int, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, float, float, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, double, double, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int, int, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, float, float, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, double, double, math::DividesFunctor);
DEFINE_BINARY_FUNC(Pow, float, float, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, double, double, math::PowFunctor);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int, int, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, float, float, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, double, double, math::MinFunctor);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int, int, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, float, float, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, double, double, math::MaxFunctor);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, float, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, double, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, float, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, double, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(Less, int8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int64_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, float, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, double, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, float, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, double, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, float, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, double, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool, math::GreaterEqualFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, T, dtype) \
  template <>                              \
  DRAGON_API void name<T, CUDAContext>(    \
      const int a_ndim,                    \
      const int64_t* a_dims,               \
      const int b_ndim,                    \
      const int64_t* b_dims,               \
      const T* a,                          \
      const T* b,                          \
      T* y,                                \
      CUDAContext* ctx) {                  \
    name(                                  \
        a_ndim,                            \
        a_dims,                            \
        b_ndim,                            \
        b_dims,                            \
        reinterpret_cast<const dtype*>(a), \
        reinterpret_cast<const dtype*>(b), \
        reinterpret_cast<dtype*>(y),       \
        ctx);                              \
  }

DEFINE_BINARY_FUNC(Add, bool, uint8_t); // Or
DEFINE_BINARY_FUNC(Sub, bool, uint8_t); // Xor
DEFINE_BINARY_FUNC(Mul, bool, uint8_t); // And
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, TOut1, TOut2, Functor)                       \
  template <>                                                                 \
  DRAGON_API void name<float16, CUDAContext>(                                 \
      const int a_ndim,                                                       \
      const int64_t* a_dims,                                                  \
      const int b_ndim,                                                       \
      const int64_t* b_dims,                                                  \
      const float16* a,                                                       \
      const float16* b,                                                       \
      TOut1* y,                                                               \
      CUDAContext* ctx) {                                                     \
    int rows, cols, broadcast_1st;                                            \
    vec64_t A_dims(a_dims, a_dims + a_ndim);                                  \
    vec64_t B_dims(b_dims, b_dims + b_ndim);                                  \
    vec64_t A_broadcast_dims, B_broadcast_dims;                               \
    utils::math::ComputeBinaryBroadcastDims(                                  \
        A_dims, B_dims, A_broadcast_dims, B_broadcast_dims);                  \
    if (A_broadcast_dims == B_broadcast_dims) {                               \
      auto count = std::accumulate(                                           \
          a_dims, a_dims + a_ndim, 1, std::multiplies<int64_t>());            \
      name(count, a, b, y, ctx);                                              \
      return;                                                                 \
    }                                                                         \
    if (utils::math::IsRowwiseBroadcast(                                      \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {                  \
      auto nthreads = rows * cols;                                            \
      if (broadcast_1st > 0) {                                                \
        _RowwiseBinaryFunc<half, TOut2, Functor<half>, true>                  \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads,                                                     \
                cols,                                                         \
                Functor<half>(),                                              \
                reinterpret_cast<const half*>(a),                             \
                reinterpret_cast<const half*>(b),                             \
                reinterpret_cast<TOut2*>(y));                                 \
      } else {                                                                \
        _RowwiseBinaryFunc<half, TOut2, Functor<half>, false>                 \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads,                                                     \
                cols,                                                         \
                Functor<half>(),                                              \
                reinterpret_cast<const half*>(a),                             \
                reinterpret_cast<const half*>(b),                             \
                reinterpret_cast<TOut2*>(y));                                 \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
    if (utils::math::IsColwiseBroadcast(                                      \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {                  \
      auto nthreads = rows * cols;                                            \
      if (broadcast_1st > 0) {                                                \
        _ColwiseBinaryFunc<half, TOut2, Functor<half>, true>                  \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads,                                                     \
                cols,                                                         \
                Functor<half>(),                                              \
                reinterpret_cast<const half*>(a),                             \
                reinterpret_cast<const half*>(b),                             \
                reinterpret_cast<TOut2*>(y));                                 \
      } else {                                                                \
        _ColwiseBinaryFunc<half, TOut2, Functor<half>, false>                 \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads,                                                     \
                cols,                                                         \
                Functor<half>(),                                              \
                reinterpret_cast<const half*>(a),                             \
                reinterpret_cast<const half*>(b),                             \
                reinterpret_cast<TOut2*>(y));                                 \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
    vec64_t A_broadcast_strides, B_broadcast_strides, Y_dims;                 \
    utils::math::ComputeBinaryBroadcastStrides(                               \
        A_dims, B_dims, A_broadcast_strides, B_broadcast_strides, Y_dims);    \
    CUDA_TENSOR_DIMS_CHECK((int)Y_dims.size());                               \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> a_strides, b_strides, y_dims;      \
    const auto nthreads = std::accumulate(                                    \
        Y_dims.begin(), Y_dims.end(), 1, std::multiplies<int64_t>());         \
    for (int i = 0; i < Y_dims.size(); ++i) {                                 \
      a_strides.data[i] = A_broadcast_strides[i];                             \
      b_strides.data[i] = B_broadcast_strides[i];                             \
      y_dims.data[i] = Y_dims[i];                                             \
    }                                                                         \
    _BroadcastBinaryFunc<half, TOut2, Functor<half>, CUDA_TENSOR_MAX_DIMS>    \
        <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
            nthreads,                                                         \
            Y_dims.size(),                                                    \
            a_strides,                                                        \
            b_strides,                                                        \
            y_dims,                                                           \
            Functor<half>(),                                                  \
            reinterpret_cast<const half*>(a),                                 \
            reinterpret_cast<const half*>(b),                                 \
            reinterpret_cast<TOut2*>(y));                                     \
  }

DEFINE_BINARY_FUNC(Add, float16, half, math::PlusFunctor);
DEFINE_BINARY_FUNC(Sub, float16, half, math::MinusFunctor);
DEFINE_BINARY_FUNC(Mul, float16, half, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Div, float16, half, math::DividesFunctor);
DEFINE_BINARY_FUNC(Pow, float16, half, math::PowFunctor);
DEFINE_BINARY_FUNC(Minimum, float16, half, math::MinFunctor);
DEFINE_BINARY_FUNC(Maximum, float16, half, math::MaxFunctor);
DEFINE_BINARY_FUNC(Equal, bool, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, bool, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(Less, bool, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(LessEqual, bool, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(Greater, bool, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool, math::GreaterEqualFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_WHERE_FUNC(T1, T2)                                           \
  template <>                                                               \
  DRAGON_API void Where<T1, CUDAContext>(                                   \
      const int a_ndim,                                                     \
      const int64_t* a_dims,                                                \
      const int b_ndim,                                                     \
      const int64_t* b_dims,                                                \
      const int c_ndim,                                                     \
      const int64_t* c_dims,                                                \
      const T1* a,                                                          \
      const T1* b,                                                          \
      const bool* c,                                                        \
      T1* y,                                                                \
      CUDAContext* ctx) {                                                   \
    vec64_t A_dims(a_dims, a_dims + a_ndim);                                \
    vec64_t B_dims(b_dims, b_dims + b_ndim);                                \
    vec64_t C_dims(c_dims, c_dims + c_ndim);                                \
    vec64_t A_broadcast_dims, B_broadcast_dims, C_broadcast_dims;           \
    vec64_t A_broadcast_strides, B_broadcast_strides, C_broadcast_strides;  \
    vec64_t Y_dims, _, __;                                                  \
    utils::math::ComputeBinaryBroadcastStrides(A_dims, B_dims, _, _, __);   \
    utils::math::ComputeBinaryBroadcastStrides(C_dims, __, _, _, Y_dims);   \
    utils::math::ComputeBinaryBroadcastDims(                                \
        A_dims, Y_dims, A_broadcast_dims, _);                               \
    utils::math::ComputeBinaryBroadcastDims(                                \
        B_dims, Y_dims, B_broadcast_dims, _);                               \
    utils::math::ComputeBinaryBroadcastDims(                                \
        C_dims, Y_dims, C_broadcast_dims, _);                               \
    if (A_broadcast_dims == B_broadcast_dims &&                             \
        B_broadcast_dims == C_broadcast_dims) {                             \
      auto count = std::accumulate(                                         \
          a_dims, a_dims + a_ndim, 1, std::multiplies<int64_t>());          \
      Where(count, a, b, c, y, ctx);                                        \
      return;                                                               \
    }                                                                       \
    CUDA_TENSOR_DIMS_CHECK((int)Y_dims.size());                             \
    utils::math::ComputeBinaryBroadcastStrides(                             \
        A_dims, Y_dims, A_broadcast_strides, _, _);                         \
    utils::math::ComputeBinaryBroadcastStrides(                             \
        B_dims, Y_dims, B_broadcast_strides, _, _);                         \
    utils::math::ComputeBinaryBroadcastStrides(                             \
        C_dims, Y_dims, C_broadcast_strides, _, _);                         \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> a_strides, b_strides, c_strides; \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> y_dims;                          \
    const auto nthreads = std::accumulate(                                  \
        Y_dims.begin(), Y_dims.end(), 1, std::multiplies<int64_t>());       \
    for (int i = 0; i < Y_dims.size(); ++i) {                               \
      a_strides.data[i] = A_broadcast_strides[i];                           \
      b_strides.data[i] = B_broadcast_strides[i];                           \
      c_strides.data[i] = C_broadcast_strides[i];                           \
      y_dims.data[i] = Y_dims[i];                                           \
    }                                                                       \
    _BroadcastWhere<<<                                                      \
        CUDA_BLOCKS(nthreads),                                              \
        CUDA_THREADS,                                                       \
        0,                                                                  \
        ctx->cuda_stream()>>>(                                              \
        nthreads,                                                           \
        Y_dims.size(),                                                      \
        a_strides,                                                          \
        b_strides,                                                          \
        c_strides,                                                          \
        y_dims,                                                             \
        reinterpret_cast<const T2*>(a),                                     \
        reinterpret_cast<const T2*>(b),                                     \
        reinterpret_cast<const uint8_t*>(c),                                \
        reinterpret_cast<T2*>(y));                                          \
  }

DEFINE_WHERE_FUNC(bool, uint8_t);
DEFINE_WHERE_FUNC(int8_t, int8_t);
DEFINE_WHERE_FUNC(uint8_t, uint8_t);
DEFINE_WHERE_FUNC(int, int);
DEFINE_WHERE_FUNC(int64_t, int64_t);
DEFINE_WHERE_FUNC(float16, half);
DEFINE_WHERE_FUNC(float, float);
DEFINE_WHERE_FUNC(double, double);
#undef DEFINE_WHERE_FUNC

} // namespace math

} // namespace dragon

#endif // USE_CUDA
