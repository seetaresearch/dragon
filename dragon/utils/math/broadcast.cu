#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/broadcast.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

/*!
 * Op Wrappers
 */

#define DEFINE_BINARY_OPERATOR(name, TOut, expr)                      \
  template <typename T>                                               \
  struct name##Op {                                                   \
    inline __device__ TOut operator()(const T& a, const T& b) const { \
      return a expr b;                                                \
    }                                                                 \
  }

DEFINE_BINARY_OPERATOR(Add, T, +);
DEFINE_BINARY_OPERATOR(Sub, T, -);
DEFINE_BINARY_OPERATOR(Mul, T, *);
DEFINE_BINARY_OPERATOR(Div, T, /);
DEFINE_BINARY_OPERATOR(Equal, bool, ==);
DEFINE_BINARY_OPERATOR(NotEqual, bool, !=);
DEFINE_BINARY_OPERATOR(Less, bool, <);
DEFINE_BINARY_OPERATOR(LessEqual, bool, <=);
DEFINE_BINARY_OPERATOR(Greater, bool, >);
DEFINE_BINARY_OPERATOR(GreaterEqual, bool, >=);
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, func)                         \
  template <typename T>                                            \
  struct name##Op {                                                \
    inline __device__ T operator()(const T& a, const T& b) const { \
      return func(a, b);                                           \
    }                                                              \
  }

DEFINE_BINARY_OPERATOR(Pow, pow);
DEFINE_BINARY_OPERATOR(Min, min);
DEFINE_BINARY_OPERATOR(Max, max);
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, TOut, func)                      \
  template <typename T>                                               \
  struct name##Op {                                                   \
    inline __device__ TOut operator()(const T& a, const T& b) const { \
      return func(a, b);                                              \
    }                                                                 \
  }

#if __CUDA_ARCH__ >= 530
DEFINE_BINARY_OPERATOR(AddHalf, T, __hadd);
DEFINE_BINARY_OPERATOR(SubHalf, T, __hsub);
DEFINE_BINARY_OPERATOR(MulHalf, T, __hmul);
DEFINE_BINARY_OPERATOR(DivHalf, T, __hdiv);
DEFINE_BINARY_OPERATOR(EqualHalf, T, __heq);
DEFINE_BINARY_OPERATOR(NotEqualHalf, T, __hne);
DEFINE_BINARY_OPERATOR(LessHalf, T, __hlt);
DEFINE_BINARY_OPERATOR(LessEqualHalf, T, __hle);
DEFINE_BINARY_OPERATOR(GreaterHalf, T, __hgt);
DEFINE_BINARY_OPERATOR(GreaterEqualHalf, T, __hge);
#endif
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, expr)                         \
  template <typename T>                                            \
  struct name##Op {                                                \
    inline __device__ T operator()(const T& a, const T& b) const { \
      return __float2half(__half2float(a) expr __half2float(b));   \
    }                                                              \
  }

#if __CUDA_ARCH__ < 530
DEFINE_BINARY_OPERATOR(AddHalf, +);
DEFINE_BINARY_OPERATOR(SubHalf, -);
DEFINE_BINARY_OPERATOR(MulHalf, *);
DEFINE_BINARY_OPERATOR(DivHalf, /);
#endif
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, func)                         \
  template <typename T>                                            \
  struct name##Op {                                                \
    inline __device__ T operator()(const T& a, const T& b) const { \
      return __float2half(func(__half2float(a), __half2float(b))); \
    }                                                              \
  }

DEFINE_BINARY_OPERATOR(PowHalf, pow);
DEFINE_BINARY_OPERATOR(MinHalf, min);
DEFINE_BINARY_OPERATOR(MaxHalf, max);
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, expr)                            \
  template <typename T>                                               \
  struct name##Op {                                                   \
    inline __device__ bool operator()(const T& a, const T& b) const { \
      return __half2float(a) expr __half2float(b);                    \
    }                                                                 \
  }

#if __CUDA_ARCH__ < 530
DEFINE_BINARY_OPERATOR(EqualHalf, ==);
DEFINE_BINARY_OPERATOR(NotEqualHalf, !=);
DEFINE_BINARY_OPERATOR(LessHalf, <);
DEFINE_BINARY_OPERATOR(LessEqualHalf, <=);
DEFINE_BINARY_OPERATOR(GreaterHalf, >);
DEFINE_BINARY_OPERATOR(GreaterEqualHalf, >=);
#endif
#undef DEFINE_BINARY_OPERATOR

/*!
 * Op Kernels
 */

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

#define DEFINE_BINARY_FUNC(name, TIn, TOut, Op)                               \
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
        _RowwiseBinaryFunc<TIn, TOut, Op<TIn>, true>                          \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads, cols, Op<TIn>(), a, b, y);                          \
      } else {                                                                \
        _RowwiseBinaryFunc<TIn, TOut, Op<TIn>, false>                         \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads, cols, Op<TIn>(), a, b, y);                          \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
    if (utils::math::IsColwiseBroadcast(                                      \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {                  \
      const auto nthreads = rows * cols;                                      \
      if (broadcast_1st > 0) {                                                \
        _ColwiseBinaryFunc<TIn, TOut, Op<TIn>, true>                          \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads, cols, Op<TIn>(), a, b, y);                          \
      } else {                                                                \
        _ColwiseBinaryFunc<TIn, TOut, Op<TIn>, false>                         \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads, cols, Op<TIn>(), a, b, y);                          \
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
    _BroadcastBinaryFunc<TIn, TOut, Op<TIn>, CUDA_TENSOR_MAX_DIMS>            \
        <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
            nthreads,                                                         \
            Y_dims.size(),                                                    \
            a_strides,                                                        \
            b_strides,                                                        \
            y_dims,                                                           \
            Op<TIn>(),                                                        \
            a,                                                                \
            b,                                                                \
            y);                                                               \
  }

DEFINE_BINARY_FUNC(Add, int8_t, int8_t, AddOp);
DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t, AddOp);
DEFINE_BINARY_FUNC(Add, int, int, AddOp);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t, AddOp);
DEFINE_BINARY_FUNC(Add, float, float, AddOp);
DEFINE_BINARY_FUNC(Add, double, double, AddOp);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t, SubOp);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t, SubOp);
DEFINE_BINARY_FUNC(Sub, int, int, SubOp);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t, SubOp);
DEFINE_BINARY_FUNC(Sub, float, float, SubOp);
DEFINE_BINARY_FUNC(Sub, double, double, SubOp);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t, MulOp);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t, MulOp);
DEFINE_BINARY_FUNC(Mul, int, int, MulOp);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t, MulOp);
DEFINE_BINARY_FUNC(Mul, float, float, MulOp);
DEFINE_BINARY_FUNC(Mul, double, double, MulOp);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t, DivOp);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t, DivOp);
DEFINE_BINARY_FUNC(Div, int, int, DivOp);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t, DivOp);
DEFINE_BINARY_FUNC(Div, float, float, DivOp);
DEFINE_BINARY_FUNC(Div, double, double, DivOp);
DEFINE_BINARY_FUNC(Pow, float, float, PowOp);
DEFINE_BINARY_FUNC(Pow, double, double, PowOp);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t, MinOp);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t, MinOp);
DEFINE_BINARY_FUNC(Minimum, int, int, MinOp);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t, MinOp);
DEFINE_BINARY_FUNC(Minimum, float, float, MinOp);
DEFINE_BINARY_FUNC(Minimum, double, double, MinOp);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t, MaxOp);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t, MaxOp);
DEFINE_BINARY_FUNC(Maximum, int, int, MaxOp);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t, MaxOp);
DEFINE_BINARY_FUNC(Maximum, float, float, MaxOp);
DEFINE_BINARY_FUNC(Maximum, double, double, MaxOp);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, int, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, float, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, double, bool, EqualOp);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, int, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, float, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, double, bool, NotEqualOp);
DEFINE_BINARY_FUNC(Less, int8_t, bool, LessOp);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, LessOp);
DEFINE_BINARY_FUNC(Less, int, bool, LessOp);
DEFINE_BINARY_FUNC(Less, int64_t, bool, LessOp);
DEFINE_BINARY_FUNC(Less, float, bool, LessOp);
DEFINE_BINARY_FUNC(Less, double, bool, LessOp);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, int, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, float, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, double, bool, LessEqualOp);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, int, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, float, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, double, bool, GreaterOp);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool, GreaterEqualOp);
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

#define DEFINE_BINARY_FUNC(name, TOut1, TOut2, Op)                            \
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
        _RowwiseBinaryFunc<half, TOut2, Op<half>, true>                       \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads,                                                     \
                cols,                                                         \
                Op<half>(),                                                   \
                reinterpret_cast<const half*>(a),                             \
                reinterpret_cast<const half*>(b),                             \
                reinterpret_cast<TOut2*>(y));                                 \
      } else {                                                                \
        _RowwiseBinaryFunc<half, TOut2, Op<half>, false>                      \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads,                                                     \
                cols,                                                         \
                Op<half>(),                                                   \
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
        _ColwiseBinaryFunc<half, TOut2, Op<half>, true>                       \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads,                                                     \
                cols,                                                         \
                Op<half>(),                                                   \
                reinterpret_cast<const half*>(a),                             \
                reinterpret_cast<const half*>(b),                             \
                reinterpret_cast<TOut2*>(y));                                 \
      } else {                                                                \
        _ColwiseBinaryFunc<half, TOut2, Op<half>, false>                      \
            <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
                nthreads,                                                     \
                cols,                                                         \
                Op<half>(),                                                   \
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
    _BroadcastBinaryFunc<half, TOut2, Op<half>, CUDA_TENSOR_MAX_DIMS>         \
        <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
            nthreads,                                                         \
            Y_dims.size(),                                                    \
            a_strides,                                                        \
            b_strides,                                                        \
            y_dims,                                                           \
            Op<half>(),                                                       \
            reinterpret_cast<const half*>(a),                                 \
            reinterpret_cast<const half*>(b),                                 \
            reinterpret_cast<TOut2*>(y));                                     \
  }

DEFINE_BINARY_FUNC(Add, float16, half, AddHalfOp);
DEFINE_BINARY_FUNC(Sub, float16, half, SubHalfOp);
DEFINE_BINARY_FUNC(Mul, float16, half, MulHalfOp);
DEFINE_BINARY_FUNC(Div, float16, half, DivHalfOp);
DEFINE_BINARY_FUNC(Pow, float16, half, PowHalfOp);
DEFINE_BINARY_FUNC(Minimum, float16, half, MinHalfOp);
DEFINE_BINARY_FUNC(Maximum, float16, half, MaxHalfOp);
DEFINE_BINARY_FUNC(Equal, bool, bool, EqualHalfOp);
DEFINE_BINARY_FUNC(NotEqual, bool, bool, NotEqualHalfOp);
DEFINE_BINARY_FUNC(Less, bool, bool, LessHalfOp);
DEFINE_BINARY_FUNC(LessEqual, bool, bool, LessEqualHalfOp);
DEFINE_BINARY_FUNC(Greater, bool, bool, GreaterHalfOp);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool, GreaterEqualHalfOp);
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