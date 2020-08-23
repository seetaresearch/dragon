#ifdef USE_CUDA

#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math/reduce.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

template <typename T, class Reducer>
__global__ void _RowwiseReduce(
    const int rows,
    const int cols,
    const Reducer reducer,
    const T init,
    const T scale,
    const T* x,
    T* y) {
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, cols) {
    T val = init;
    CUDA_2D_KERNEL_LOOP2(j, rows) {
      val = reducer(val, x[j * cols + i]);
    }
    val = BlockReduce<T>(storage).Reduce(val, reducer);
    if (threadIdx.x == 0) y[i] = val * scale;
  }
}

template <class Reducer>
__global__ void _RowwiseReduce(
    const int rows,
    const int cols,
    const Reducer reducer,
    const float init,
    const float scale,
    const half* x,
    half* y) {
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, cols) {
    float val = init;
    CUDA_2D_KERNEL_LOOP2(j, rows) {
      val = reducer(val, __half2float(x[j * cols + i]));
    }
    val = BlockReduce<float>(storage).Reduce(val, reducer);
    if (threadIdx.x == 0) y[i] = __float2half(val * scale);
  }
}

template <typename T, class Reducer>
__global__ void _ColwiseReduce(
    const int rows,
    const int cols,
    const Reducer reducer,
    const T init,
    const T scale,
    const T* x,
    T* y) {
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    T val = init;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      val = reducer(val, x[i * cols + j]);
    }
    val = BlockReduce<T>(storage).Reduce(val, reducer);
    if (threadIdx.x == 0) y[i] = val * scale;
  }
}

template <class Reducer>
__global__ void _ColwiseReduce(
    const int rows,
    const int cols,
    const Reducer reducer,
    const float init,
    const float scale,
    const half* x,
    half* y) {
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    float val = init;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      val = reducer(val, __half2float(x[i * cols + j]));
    }
    val = BlockReduce<float>(storage).Reduce(val, reducer);
    if (threadIdx.x == 0) y[i] = __float2half(val * scale);
  }
}

template <typename T, class Reducer, int D>
__global__ void _GenericReduce(
    const int rows,
    const int cols,
    const int num_dims,
    const SimpleArray<int, D> x_dims,
    const SimpleArray<int, D> x_strides,
    const Reducer reducer,
    const T init,
    const T scale,
    const T* x,
    T* y) {
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    T val = init;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      int xi = 0, c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(x_dims.data[d], c, &c, &r);
        xi += r * x_strides.data[d];
      }
      val = reducer(val, x[xi]);
    }
    val = BlockReduce<T>(storage).Reduce(val, reducer);
    if (threadIdx.x == 0) y[i] = val * scale;
  }
}

template <class Reducer, int D>
__global__ void _GenericReduce(
    const int rows,
    const int cols,
    const int num_dims,
    const SimpleArray<int, D> x_dims,
    const SimpleArray<int, D> x_strides,
    const Reducer reducer,
    const float init,
    const float scale,
    const half* x,
    half* y) {
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    float val = init;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      int xi = 0, c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(x_dims.data[d], c, &c, &r);
        xi += r * x_strides.data[d];
      }
      val = reducer(val, __half2float(x[xi]));
    }
    val = BlockReduce<float>(storage).Reduce(val, reducer);
    if (threadIdx.x == 0) y[i] = __float2half(val * scale);
  }
}

#define DEFINE_REDUCE_FUNCTION(name)                                           \
  template <typename Tx, typename Tp, class Reducer>                           \
  void _Reduce##name(                                                          \
      const int num_dims,                                                      \
      const int* dims,                                                         \
      const int num_axes,                                                      \
      const int* axes,                                                         \
      const Reducer reducer,                                                   \
      const Tp init,                                                           \
      const Tp scale,                                                          \
      const Tx* x,                                                             \
      Tx* y,                                                                   \
      CUDAContext* ctx) {                                                      \
    int rows, cols;                                                            \
    vec32_t y_dims(dims, dims + num_dims);                                     \
    for (int i = 0; i < num_axes; ++i)                                         \
      y_dims[axes[i]] = 1;                                                     \
    /*! Case #1: Rowwise Reduce */                                             \
    if (utils::math::IsRowwiseReduce(                                          \
            num_dims, dims, y_dims.data(), &rows, &cols)) {                    \
      _RowwiseReduce<<<                                                        \
          CUDA_2D_BLOCKS(cols),                                                \
          CUDA_THREADS,                                                        \
          0,                                                                   \
          ctx->cuda_stream()>>>(rows, cols, reducer, init, scale, x, y);       \
      return;                                                                  \
    }                                                                          \
    /*! Case #2: Colwise Reduce */                                             \
    if (utils::math::IsColwiseReduce(                                          \
            num_dims, dims, y_dims.data(), &rows, &cols)) {                    \
      _ColwiseReduce<<<                                                        \
          CUDA_2D_BLOCKS(rows),                                                \
          CUDA_THREADS,                                                        \
          0,                                                                   \
          ctx->cuda_stream()>>>(rows, cols, reducer, init, scale, x, y);       \
      return;                                                                  \
    }                                                                          \
    /*! Case #3: Generic Reduce */                                             \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                          \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> axesT, stridesT, dimsT;             \
    utils::math::TransposeAxesForReduce(num_dims, num_axes, axes, axesT.data); \
    utils::math::ComputeTransposeStrides(                                      \
        num_dims, dims, axesT.data, stridesT.data);                            \
    rows = cols = 1;                                                           \
    const int pivot = num_dims - num_axes;                                     \
    for (int i = 0; i < pivot; ++i)                                            \
      rows *= dims[axesT.data[i]];                                             \
    for (int i = pivot; i < num_dims; ++i)                                     \
      cols *= dims[axesT.data[i]];                                             \
    for (int i = 0; i < num_dims; ++i)                                         \
      dimsT.data[i] = dims[axesT.data[i]];                                     \
    _GenericReduce<<<                                                          \
        CUDA_2D_BLOCKS(rows),                                                  \
        CUDA_THREADS,                                                          \
        0,                                                                     \
        ctx->cuda_stream()>>>(                                                 \
        rows, cols, num_dims, dimsT, stridesT, reducer, init, scale, x, y);    \
  }

DEFINE_REDUCE_FUNCTION(Max);
DEFINE_REDUCE_FUNCTION(Min);
DEFINE_REDUCE_FUNCTION(Sum);
#undef DEFINE_REDUCE_FUNCTION

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ReduceMax<float16, CUDAContext>(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _ReduceMax(
      num_dims,
      dims,
      num_axes,
      axes,
      cub::Max(),
      std::numeric_limits<float>::lowest(),
      1.f,
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(y),
      ctx);
}

template <>
void ReduceMin<float16, CUDAContext>(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _ReduceMin(
      num_dims,
      dims,
      num_axes,
      axes,
      cub::Min(),
      std::numeric_limits<float>::max(),
      1.f,
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(y),
      ctx);
}

template <>
void ReduceSum<float16, CUDAContext>(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const float scale,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _ReduceMin(
      num_dims,
      dims,
      num_axes,
      axes,
      cub::Sum(),
      0.f,
      scale,
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(y),
      ctx);
}

#define DEFINE_KERNEL_LAUNCHER(name, T, Reducer, kInit)                     \
  template <>                                                               \
  void Reduce##name<T, CUDAContext>(                                        \
      const int num_dims,                                                   \
      const int* dims,                                                      \
      const int num_axes,                                                   \
      const int* axes,                                                      \
      const T* x,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    _Reduce##name(                                                          \
        num_dims, dims, num_axes, axes, Reducer(), kInit, T(1), x, y, ctx); \
  }

DEFINE_KERNEL_LAUNCHER(
    Max,
    int8_t,
    cub::Max,
    std::numeric_limits<int8_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Max,
    uint8_t,
    cub::Max,
    std::numeric_limits<uint8_t>::lowest());
DEFINE_KERNEL_LAUNCHER(Max, int, cub::Max, std::numeric_limits<int>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Max,
    int64_t,
    cub::Max,
    std::numeric_limits<int64_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Max,
    float,
    cub::Max,
    std::numeric_limits<float>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Max,
    double,
    cub::Max,
    std::numeric_limits<double>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Min,
    int8_t,
    cub::Min,
    std::numeric_limits<int8_t>::max());
DEFINE_KERNEL_LAUNCHER(
    Min,
    uint8_t,
    cub::Min,
    std::numeric_limits<uint8_t>::max());
DEFINE_KERNEL_LAUNCHER(Min, int, cub::Min, std::numeric_limits<int>::max());
DEFINE_KERNEL_LAUNCHER(
    Min,
    int64_t,
    cub::Min,
    std::numeric_limits<int64_t>::max());
DEFINE_KERNEL_LAUNCHER(Min, float, cub::Min, std::numeric_limits<float>::max());
DEFINE_KERNEL_LAUNCHER(
    Min,
    double,
    cub::Min,
    std::numeric_limits<double>::max());
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T, Reducer, kInit) \
  template <>                                           \
  void Reduce##name<T, CUDAContext>(                    \
      const int num_dims,                               \
      const int* dims,                                  \
      const int num_axes,                               \
      const int* axes,                                  \
      const float scale,                                \
      const T* x,                                       \
      T* y,                                             \
      CUDAContext* ctx) {                               \
    _Reduce##name(                                      \
        num_dims,                                       \
        dims,                                           \
        num_axes,                                       \
        axes,                                           \
        Reducer(),                                      \
        kInit,                                          \
        (T)scale,                                       \
        x,                                              \
        y,                                              \
        ctx);                                           \
  }

DEFINE_KERNEL_LAUNCHER(Sum, int8_t, cub::Sum, int8_t(0));
DEFINE_KERNEL_LAUNCHER(Sum, uint8_t, cub::Sum, uint8_t(0));
DEFINE_KERNEL_LAUNCHER(Sum, int, cub::Sum, int(0));
DEFINE_KERNEL_LAUNCHER(Sum, int64_t, cub::Sum, int64_t(0));
DEFINE_KERNEL_LAUNCHER(Sum, float, cub::Sum, 0.f);
DEFINE_KERNEL_LAUNCHER(Sum, double, cub::Sum, 0.);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_SUM_FUNC(T)                                                  \
  template <>                                                               \
  DRAGON_API void Sum<T, CUDAContext>(                                      \
      const int n, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    vec32_t dims = {n}, axes = {0};                                         \
    ReduceSum(1, dims.data(), 1, axes.data(), alpha, x, y, ctx);            \
  }                                                                         \
  template <>                                                               \
  DRAGON_API T Sum<T, CUDAContext>(                                         \
      const int n, const float alpha, const T* x, CUDAContext* ctx) {       \
    T val, *y = (T*)ctx->New(sizeof(T));                                    \
    Sum(n, alpha, x, y, ctx);                                               \
    CUDA_CHECK(cudaMemcpyAsync(                                             \
        &val, y, sizeof(T), cudaMemcpyDeviceToHost, ctx->cuda_stream()));   \
    ctx->FinishDeviceComputation();                                         \
    ctx->Delete(y);                                                         \
    return val;                                                             \
  }

DEFINE_SUM_FUNC(int8_t);
DEFINE_SUM_FUNC(uint8_t);
DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float16);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

} // namespace math

} // namespace dragon

#endif // USE_CUDA
