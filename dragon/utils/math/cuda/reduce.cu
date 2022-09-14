#include "dragon/core/workspace.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/reduce.h"
#include "dragon/utils/math/transpose.h"
#include "dragon/utils/math/types.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

template <typename T, typename AccT, class Functor, class Reducer>
__global__ void _RowwiseReduce(
    const int rows,
    const int cols,
    const Functor op,
    const Reducer reducer,
    const AccT init,
    const AccT scale,
    const T* x,
    T* y) {
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, cols) {
    AccT val = init;
    CUDA_2D_KERNEL_LOOP2(j, rows) {
      val = reducer(val, op(x[j * cols + i]));
    }
    val = BlockReduce<AccT>(storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      y[i] = convert::To<T>(val * scale);
    }
  }
}

template <typename T, typename AccT, class Functor, class Reducer>
__global__ void _ColwiseReduce(
    const int rows,
    const int cols,
    const Functor op,
    const Reducer reducer,
    const AccT init,
    const AccT scale,
    const T* x,
    T* y) {
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    AccT val = init;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      val = reducer(val, op(x[i * cols + j]));
    }
    val = BlockReduce<AccT>(storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      y[i] = convert::To<T>(val * scale);
    }
  }
}

template <typename T, typename AccT, class Functor, class Reducer, int D>
__global__ void _GenericReduce(
    const int rows,
    const int cols,
    const SimpleArray<int, D> x_dims,
    const SimpleArray<int, D> x_strides,
    const Functor op,
    const Reducer reducer,
    const AccT init,
    const AccT scale,
    const T* x,
    T* y) {
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    AccT val = init;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      int xi = 0, c = i * cols + j;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(x_dims.data[d], c, &c, &r);
        xi += r * x_strides.data[d];
      }
      val = reducer(val, op(x[xi]));
    }
    val = BlockReduce<AccT>(storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      y[i] = convert::To<T>(val * scale);
    }
  }
}

template <class Functor, class Reducer, int D>
void _GenericReduceImpl(
    const int* dims,
    const int num_axes,
    const int* axes,
    const typename Functor::OutputT init,
    const typename Functor::OutputT scale,
    const typename Functor::InputT* x,
    typename Functor::InputT* y,
    CUDAContext* ctx) {
  SimpleArray<int, D> transpose_axes;
  SimpleArray<int, D> transpose_strides;
  SimpleArray<int, D> transpose_dims;
  math::utils::TransposeAxesForReduce(D, num_axes, axes, transpose_axes.data);
  math::utils::ComputeTransposeStrides(
      D, dims, transpose_axes.data, transpose_strides.data);
  int rows = 1, cols = 1;
  const int pivot = D - num_axes;
  for (int i = 0; i < pivot; ++i) {
    rows *= dims[transpose_axes.data[i]];
  }
  for (int i = pivot; i < D; ++i) {
    cols *= dims[transpose_axes.data[i]];
  }
  for (int i = 0; i < D; ++i) {
    transpose_dims.data[i] = dims[transpose_axes.data[i]];
  }
  _GenericReduce<<<rows, CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      rows,
      cols,
      transpose_dims,
      transpose_strides,
      Functor(),
      Reducer(),
      init,
      scale,
      x,
      y);
}

template <typename T, typename AccT, class Functor, class Reducer>
void _DeviceReduceImpl(
    const int N,
    const AccT init,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  size_t buffer_size = 0;
  cub::TransformInputIterator<AccT, Functor, const T*> X(x, Functor());
  cub::CacheModifiedOutputIterator<cub::STORE_DEFAULT, T> Y(y);
  cub::DeviceReduce::Reduce(
      nullptr, buffer_size, X, Y, N, Reducer(), init, ctx->cuda_stream());
  cub::DeviceReduce::Reduce(
      ctx->workspace()->data<CUDAContext>(buffer_size, "BufferKernel"),
      buffer_size,
      X,
      Y,
      N,
      Reducer(),
      init,
      ctx->cuda_stream());
}

template <typename T, typename AccT, class Functor>
struct ReduceFunctor {
  typedef T InputT;
  typedef AccT OutputT;
  inline __device__ AccT operator()(const T& x) const {
    return convert::To<AccT>(functor_(x));
  }
  Functor functor_;
};

#define DEFINE_REDUCE_DISPATCHER(name, Functor, Reducer)                       \
  template <typename T, typename AccT>                                         \
  int _Reduce##name(                                                           \
      const int num_dims,                                                      \
      const int64_t* dims,                                                     \
      const int num_axes,                                                      \
      const int64_t* axes,                                                     \
      const AccT init,                                                         \
      const AccT scale,                                                        \
      const T* x,                                                              \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    using FunctorT = ReduceFunctor<T, AccT, Functor<T>>;                       \
    const auto N = math::utils::Prod(num_dims, dims);                          \
    if (num_dims == num_axes && N > 10000) {                                   \
      _DeviceReduceImpl<T, AccT, FunctorT, Reducer<AccT>>(N, init, x, y, ctx); \
      return 0;                                                                \
    }                                                                          \
    int64_t rows, cols;                                                        \
    vec64_t out_dims(dims, dims + num_dims);                                   \
    for (int i = 0; i < num_axes; ++i) {                                       \
      out_dims[axes[i]] = 1;                                                   \
    }                                                                          \
    if (math::utils::IsRowwiseReduce(                                          \
            num_dims, dims, out_dims.data(), &rows, &cols)) {                  \
      _RowwiseReduce<<<cols, CUDA_THREADS, 0, ctx->cuda_stream()>>>(           \
          rows, cols, FunctorT(), Reducer<AccT>(), init, scale, x, y);         \
      return 1;                                                                \
    }                                                                          \
    if (math::utils::IsColwiseReduce(                                          \
            num_dims, dims, out_dims.data(), &rows, &cols)) {                  \
      _ColwiseReduce<<<rows, CUDA_THREADS, 0, ctx->cuda_stream()>>>(           \
          rows, cols, FunctorT(), Reducer<AccT>(), init, scale, x, y);         \
      return 2;                                                                \
    }                                                                          \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                          \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_2(                                        \
        _GenericReduceImpl,                                                    \
        FunctorT,                                                              \
        Reducer<AccT>,                                                         \
        num_dims,                                                              \
        vec32_t(dims, dims + num_dims).data(),                                 \
        num_axes,                                                              \
        vec32_t(axes, axes + num_axes).data(),                                 \
        init,                                                                  \
        scale,                                                                 \
        x,                                                                     \
        y,                                                                     \
        ctx);                                                                  \
    return 3;                                                                  \
  }

DEFINE_REDUCE_DISPATCHER(Max, math::IdentityFunctor, math::MaxFunctor);
DEFINE_REDUCE_DISPATCHER(Min, math::IdentityFunctor, math::MinFunctor);
DEFINE_REDUCE_DISPATCHER(Sum, math::IdentityFunctor, math::PlusFunctor);
DEFINE_REDUCE_DISPATCHER(SumSqr, math::SqrFunctor, math::PlusFunctor);
DEFINE_REDUCE_DISPATCHER(L1, math::AbsFunctor, math::PlusFunctor);
#undef DEFINE_REDUCE_DISPATCHER

} // namespace

#define DEFINE_REDUCE_FUNC(name, T, AccT, kInit)               \
  template <>                                                  \
  DRAGON_API void Reduce##name<T, CUDAContext>(                \
      const int num_dims,                                      \
      const int64_t* dims,                                     \
      const int num_axes,                                      \
      const int64_t* axes,                                     \
      const float scale,                                       \
      const T* x,                                              \
      T* y,                                                    \
      CUDAContext* ctx) {                                      \
    vec64_t new_dims, new_axes;                                \
    math::utils::CollapseReduceAxes(                           \
        num_dims, dims, num_axes, axes, new_dims, new_axes);   \
    auto reduce_type = _Reduce##name(                          \
        new_dims.size(),                                       \
        new_dims.data(),                                       \
        new_axes.size(),                                       \
        new_axes.data(),                                       \
        convert::To<AccT>(kInit),                              \
        convert::To<AccT>(scale),                              \
        reinterpret_cast<const math::ScalarType<T>::type*>(x), \
        reinterpret_cast<math::ScalarType<T>::type*>(y),       \
        ctx);                                                  \
    if (reduce_type == 0 && scale != 1.f) {                    \
      math::Scale(1, scale, y, y, ctx);                        \
    }                                                          \
  }

DEFINE_REDUCE_FUNC(
    Max,
    uint8_t,
    uint8_t,
    std::numeric_limits<uint8_t>::lowest());
DEFINE_REDUCE_FUNC(Max, int8_t, int8_t, std::numeric_limits<int8_t>::lowest());
DEFINE_REDUCE_FUNC(Max, int, int, std::numeric_limits<int>::lowest());
DEFINE_REDUCE_FUNC(
    Max,
    int64_t,
    int64_t,
    std::numeric_limits<int64_t>::lowest());
DEFINE_REDUCE_FUNC(Max, float16, float, cub::Traits<half>::Lowest());
DEFINE_REDUCE_FUNC(Max, float, float, std::numeric_limits<float>::lowest());
DEFINE_REDUCE_FUNC(Max, double, double, std::numeric_limits<double>::lowest());
DEFINE_REDUCE_FUNC(Min, uint8_t, uint8_t, std::numeric_limits<uint8_t>::max());
DEFINE_REDUCE_FUNC(Min, int8_t, int8_t, std::numeric_limits<int8_t>::max());
DEFINE_REDUCE_FUNC(Min, int, int, std::numeric_limits<int>::max());
DEFINE_REDUCE_FUNC(Min, int64_t, int64_t, std::numeric_limits<int64_t>::max());
DEFINE_REDUCE_FUNC(Min, float16, float, cub::Traits<half>::Max());
DEFINE_REDUCE_FUNC(Min, float, float, std::numeric_limits<float>::max());
DEFINE_REDUCE_FUNC(Min, double, double, std::numeric_limits<double>::max());
DEFINE_REDUCE_FUNC(Sum, int, int, int(0));
DEFINE_REDUCE_FUNC(Sum, int64_t, int64_t, int64_t(0));
DEFINE_REDUCE_FUNC(Sum, float16, float, 0.f);
DEFINE_REDUCE_FUNC(Sum, float, float, 0.f);
DEFINE_REDUCE_FUNC(Sum, double, double, 0.);
DEFINE_REDUCE_FUNC(SumSqr, int, int, int(0));
DEFINE_REDUCE_FUNC(SumSqr, int64_t, int64_t, int64_t(0));
DEFINE_REDUCE_FUNC(SumSqr, float16, float, 0.f);
DEFINE_REDUCE_FUNC(SumSqr, float, float, 0.f);
DEFINE_REDUCE_FUNC(SumSqr, double, double, 0.);
DEFINE_REDUCE_FUNC(L1, float16, float, 0.f);
DEFINE_REDUCE_FUNC(L1, float, float, 0.f);
DEFINE_REDUCE_FUNC(L1, double, double, 0.);
#undef DEFINE_REDUCE_FUNC

#define DEFINE_SUM_FUNC(T)                                                  \
  template <>                                                               \
  DRAGON_API void Sum<T, CUDAContext>(                                      \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    vec64_t dims = {N}, axes = {0};                                         \
    math::ReduceSum(1, dims.data(), 1, axes.data(), alpha, x, y, ctx);      \
  }

DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float16);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

#define DEFINE_SUM_FUNC(T)                                            \
  template <>                                                         \
  DRAGON_API T Sum<T, CUDAContext>(                                   \
      const int N, const float alpha, const T* x, CUDAContext* ctx) { \
    auto policy = thrust::cuda::par.on(ctx->cuda_stream());           \
    auto val = thrust::reduce(policy, x, x + N) * alpha;              \
    return static_cast<T>(val);                                       \
  }

DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

} // namespace math

} // namespace dragon
