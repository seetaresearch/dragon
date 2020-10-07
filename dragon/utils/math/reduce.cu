#ifdef USE_CUDA

#include "dragon/core/workspace.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/functional.h"
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
    if (threadIdx.x == 0) {
      y[i] = math::MultipliesFunctor<T>()(val, scale);
    }
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
    if (threadIdx.x == 0) {
      y[i] = math::MultipliesFunctor<T>()(val, scale);
    }
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
    if (threadIdx.x == 0) {
      y[i] = math::MultipliesFunctor<T>()(val, scale);
    }
  }
}

#define DEFINE_REDUCE_FUNCTION(name)                                           \
  template <typename T, typename Reducer>                                      \
  int _Reduce##name(                                                           \
      const int num_dims,                                                      \
      const int* dims,                                                         \
      const int num_axes,                                                      \
      const int* axes,                                                         \
      const Reducer reducer,                                                   \
      const T init,                                                            \
      const float scale,                                                       \
      const T* x,                                                              \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    const int count =                                                          \
        std::accumulate(dims, dims + num_dims, 1, std::multiplies<int>());     \
    if (num_dims == num_axes && count > 10000) {                               \
      size_t ws_nbytes = 0;                                                    \
      cub::DeviceReduce::Reduce(                                               \
          nullptr,                                                             \
          ws_nbytes,                                                           \
          x,                                                                   \
          y,                                                                   \
          count,                                                               \
          reducer,                                                             \
          cast::to<T>(init),                                                   \
          ctx->cuda_stream());                                                 \
      cub::DeviceReduce::Reduce(                                               \
          ctx->workspace()->data<CUDAContext>({ws_nbytes})[0],                 \
          ws_nbytes,                                                           \
          x,                                                                   \
          y,                                                                   \
          count,                                                               \
          reducer,                                                             \
          cast::to<T>(init),                                                   \
          ctx->cuda_stream());                                                 \
      return 0;                                                                \
    }                                                                          \
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
          ctx->cuda_stream()>>>(                                               \
          rows, cols, reducer, init, cast::to<T>(scale), x, y);                \
      return 1;                                                                \
    }                                                                          \
    /*! Case #2: Colwise Reduce */                                             \
    if (utils::math::IsColwiseReduce(                                          \
            num_dims, dims, y_dims.data(), &rows, &cols)) {                    \
      _ColwiseReduce<<<                                                        \
          CUDA_2D_BLOCKS(rows),                                                \
          CUDA_THREADS,                                                        \
          0,                                                                   \
          ctx->cuda_stream()>>>(                                               \
          rows, cols, reducer, init, cast::to<T>(scale), x, y);                \
      return 2;                                                                \
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
        rows,                                                                  \
        cols,                                                                  \
        num_dims,                                                              \
        dimsT,                                                                 \
        stridesT,                                                              \
        reducer,                                                               \
        init,                                                                  \
        cast::to<T>(scale),                                                    \
        x,                                                                     \
        y);                                                                    \
    return 3;                                                                  \
  }

DEFINE_REDUCE_FUNCTION(Max);
DEFINE_REDUCE_FUNCTION(Min);
DEFINE_REDUCE_FUNCTION(Sum);
#undef DEFINE_REDUCE_FUNCTION

} // namespace

/* ------------------- Launcher Separator ------------------- */

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
    auto kind = _Reduce##name(                          \
        num_dims,                                       \
        dims,                                           \
        num_axes,                                       \
        axes,                                           \
        Reducer<T>(),                                   \
        kInit,                                          \
        scale,                                          \
        x,                                              \
        y,                                              \
        ctx);                                           \
    if (kind == 0) {                                    \
      math::Scale(1, scale, y, y, ctx);                 \
    }                                                   \
  }

DEFINE_KERNEL_LAUNCHER(
    Max,
    int8_t,
    math::MaxFunctor,
    std::numeric_limits<int8_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Max,
    uint8_t,
    math::MaxFunctor,
    std::numeric_limits<uint8_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Max,
    int,
    math::MaxFunctor,
    std::numeric_limits<int>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Max,
    int64_t,
    math::MaxFunctor,
    std::numeric_limits<int64_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Max,
    float16,
    math::MaxFunctor,
    cast::to<float16>(cub::Traits<half>::Lowest()));
DEFINE_KERNEL_LAUNCHER(
    Max,
    float,
    math::MaxFunctor,
    std::numeric_limits<float>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Max,
    double,
    math::MaxFunctor,
    std::numeric_limits<double>::lowest());
DEFINE_KERNEL_LAUNCHER(
    Min,
    int8_t,
    math::MinFunctor,
    std::numeric_limits<int8_t>::max());
DEFINE_KERNEL_LAUNCHER(
    Min,
    uint8_t,
    math::MinFunctor,
    std::numeric_limits<uint8_t>::max());
DEFINE_KERNEL_LAUNCHER(
    Min,
    int,
    math::MinFunctor,
    std::numeric_limits<int>::max());
DEFINE_KERNEL_LAUNCHER(
    Min,
    int64_t,
    math::MinFunctor,
    std::numeric_limits<int64_t>::max());
DEFINE_KERNEL_LAUNCHER(
    Min,
    float16,
    math::MinFunctor,
    cast::to<float16>(cub::Traits<half>::Max()));
DEFINE_KERNEL_LAUNCHER(
    Min,
    float,
    math::MinFunctor,
    std::numeric_limits<float>::max());
DEFINE_KERNEL_LAUNCHER(
    Min,
    double,
    math::MinFunctor,
    std::numeric_limits<double>::max());
DEFINE_KERNEL_LAUNCHER(Sum, int8_t, math::PlusFunctor, int8_t(0));
DEFINE_KERNEL_LAUNCHER(Sum, uint8_t, math::PlusFunctor, uint8_t(0));
DEFINE_KERNEL_LAUNCHER(Sum, int, math::PlusFunctor, int(0));
DEFINE_KERNEL_LAUNCHER(Sum, int64_t, math::PlusFunctor, int64_t(0));
DEFINE_KERNEL_LAUNCHER(Sum, float16, math::PlusFunctor, cast::to<float16>(0.f));
DEFINE_KERNEL_LAUNCHER(Sum, float, math::PlusFunctor, 0.f);
DEFINE_KERNEL_LAUNCHER(Sum, double, math::PlusFunctor, 0.);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_SUM_FUNC(T)                                                  \
  template <>                                                               \
  DRAGON_API void Sum<T, CUDAContext>(                                      \
      const int n, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    vec32_t dims = {n}, axes = {0};                                         \
    math::ReduceSum(1, dims.data(), 1, axes.data(), alpha, x, y, ctx);      \
  }

DEFINE_SUM_FUNC(int8_t);
DEFINE_SUM_FUNC(uint8_t);
DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float16);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

#define DEFINE_SUM_FUNC(T)                                            \
  template <>                                                         \
  DRAGON_API T Sum<T, CUDAContext>(                                   \
      const int n, const float alpha, const T* x, CUDAContext* ctx) { \
    auto policy = thrust::cuda::par.on(ctx->cuda_stream());           \
    auto val = thrust::reduce(policy, x, x + n) * alpha;              \
    return static_cast<T>(val);                                       \
  }

DEFINE_SUM_FUNC(int8_t);
DEFINE_SUM_FUNC(uint8_t);
DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

} // namespace math

} // namespace dragon

#endif // USE_CUDA
