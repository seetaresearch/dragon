#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
struct LessFunctorWrapper {
  LessFunctorWrapper(int64_t stride0, int64_t stride1)
      : stride0_(stride0), stride1_(stride1) {}
  inline __device__ bool operator()(
      const thrust::tuple<int64_t, T>& lhs,
      const thrust::tuple<int64_t, T>& rhs) const {
    int64_t li = thrust::get<0>(lhs), ri = thrust::get<0>(rhs);
    li = (li / stride0_) * stride1_ + li % stride1_;
    ri = (ri / stride0_) * stride1_ + ri % stride1_;
    if (li != ri) {
      return li < ri;
    } else {
      return functor_(thrust::get<1>(lhs), thrust::get<1>(rhs));
    }
  }
  int64_t stride0_, stride1_;
  math::LessFunctor<T> functor_;
};

template <typename T>
struct GreaterFunctorWrapper {
  GreaterFunctorWrapper(int64_t stride0, int64_t stride1)
      : stride0_(stride0), stride1_(stride1) {}
  inline __device__ bool operator()(
      const thrust::tuple<int64_t, T>& lhs,
      const thrust::tuple<int64_t, T>& rhs) const {
    int64_t li = thrust::get<0>(lhs), ri = thrust::get<0>(rhs);
    li = (li / stride0_) * stride1_ + li % stride1_;
    ri = (ri / stride0_) * stride1_ + ri % stride1_;
    if (li != ri) {
      return li < ri;
    } else {
      return functor_(thrust::get<1>(lhs), thrust::get<1>(rhs));
    }
  }
  int64_t stride0_, stride1_;
  math::GreaterFunctor<T> functor_;
};

template <typename T, int ItemsPerThread>
__global__ void _SelectViaBlockSort(
    const int rows,
    const int cols,
    const int inner_dim,
    const int select_dim,
    const bool largest,
    const T init,
    const T* x,
    T* y,
    int64_t* index) {
  typedef cub::BlockRadixSort<T, CUDA_THREADS, ItemsPerThread, int64_t>
      BlockSort;
  __shared__ typename BlockSort::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    T keys[ItemsPerThread];
    int64_t values[ItemsPerThread];
    const int thread_offset = threadIdx.x * ItemsPerThread;
    const int x_offset = (i / inner_dim) * cols * inner_dim + (i % inner_dim);
    const int y_offset =
        (i / inner_dim) * select_dim * inner_dim + (i % inner_dim);
#pragma unroll
    for (int j = 0; j < ItemsPerThread; ++j) {
      const int item_idx = thread_offset + j;
      values[j] = item_idx < cols ? item_idx : cols - 1;
      keys[j] = item_idx < cols ? x[x_offset + item_idx * inner_dim] : init;
    }
    __syncthreads();
    if (largest) {
      BlockSort(storage).SortDescending(keys, values);
    } else {
      BlockSort(storage).Sort(keys, values);
    }
#pragma unroll
    for (int j = 0; j < ItemsPerThread; ++j) {
      if (thread_offset + j < select_dim) {
        y[y_offset + (thread_offset + j) * inner_dim] = keys[j];
        index[y_offset + (thread_offset + j) * inner_dim] = values[j];
      }
    }
  }
}

template <typename T>
void _DeviceSort(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int largest,
    T* key,
    int64_t* value,
    CUDAContext* ctx) {
  const int rows = outer_dim * inner_dim, cols = axis_dim;
  const int count = rows * cols;
  // Fill value with global index
  auto policy = thrust::cuda::par.on(ctx->cuda_stream());
  thrust::sequence(policy, value, value + count);
  if (rows == 1) {
    // Sort a flatten array
    if (largest > 0) {
      thrust::sort_by_key(
          policy, key, key + count, value, math::GreaterFunctor<T>());
    } else {
      thrust::sort_by_key(
          policy, key, key + count, value, math::LessFunctor<T>());
    }
  } else {
    // Sort a transposed array to handle multiple rows
    auto iter = thrust::make_zip_iterator(thrust::make_tuple(value, key));
    if (largest > 0) {
      thrust::sort(
          policy,
          iter,
          iter + count,
          GreaterFunctorWrapper<T>(axis_dim * inner_dim, inner_dim));
    } else {
      thrust::sort(
          policy,
          iter,
          iter + count,
          LessFunctorWrapper<T>(axis_dim * inner_dim, inner_dim));
    }
  }
}

template <typename T>
__global__ void _SelectViaDeviceSort(
    const int nthreads,
    const int axis_dim,
    const int inner_dim,
    const int select_dim,
    const T* sorted_keys,
    const int64_t* sorted_values,
    T* y,
    int64_t* index) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int xi =
        ((yi / inner_dim / select_dim) * inner_dim + yi % inner_dim) *
            axis_dim +
        (yi / inner_dim) % select_dim;
    y[yi] = sorted_keys[xi];
    index[yi] = (sorted_values[xi] / inner_dim) % axis_dim;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define BLOCKSORT_KERNEL(T, items_per_thread)                          \
  _SelectViaBlockSort<T, items_per_thread>                             \
      <<<CUDA_2D_BLOCKS(rows), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          rows,                                                        \
          cols,                                                        \
          inner_dim,                                                   \
          select_dim,                                                  \
          largest > 0,                                                 \
          init,                                                        \
          reinterpret_cast<const T*>(x),                               \
          reinterpret_cast<T*>(value),                                 \
          index)

#define DISPATCH_BLOCKSORT_KERNEL(T)                             \
  if (cols <= CUDA_THREADS) {                                    \
    BLOCKSORT_KERNEL(T, 1);                                      \
  } else if (cols <= CUDA_THREADS * 2) {                         \
    BLOCKSORT_KERNEL(T, 2);                                      \
  } else if (cols <= CUDA_THREADS * 4) {                         \
    BLOCKSORT_KERNEL(T, 4);                                      \
  } else if (cols <= CUDA_THREADS * 8) {                         \
    BLOCKSORT_KERNEL(T, 8);                                      \
  } else {                                                       \
    LOG(FATAL) << "Too larger dimension (> " << CUDA_THREADS * 8 \
               << ") to launch the cuda kernel";                 \
  }

#define DEFINE_KERNEL_LAUNCHER(T1, T2, kLowest, kMax)                     \
  template <>                                                             \
  void TopSelect<T1, CUDAContext>(                                        \
      const int outer_dim,                                                \
      const int inner_dim,                                                \
      const int axis_dim,                                                 \
      const int select_dim,                                               \
      const int largest,                                                  \
      const T1* x,                                                        \
      T1* value,                                                          \
      int64_t* index,                                                     \
      CUDAContext* ctx) {                                                 \
    const int rows = outer_dim * inner_dim;                               \
    const int cols = axis_dim;                                            \
    if (rows == 1 || cols > CUDA_THREADS * 8) {                           \
      const int in_count = outer_dim * inner_dim * axis_dim;              \
      const int out_count = outer_dim * inner_dim * select_dim;           \
      auto data = ctx->workspace()->template data<CUDAContext>(           \
          {in_count * sizeof(T1), in_count * sizeof(int64_t)}, "data:1"); \
      math::Copy(in_count, x, (T1*)data[0], ctx);                         \
      _DeviceSort(                                                        \
          outer_dim,                                                      \
          inner_dim,                                                      \
          axis_dim,                                                       \
          largest,                                                        \
          (T1*)data[0],                                                   \
          (int64_t*)data[1],                                              \
          ctx);                                                           \
      if (rows == 1) {                                                    \
        math::Copy(out_count, (T1*)data[0], value, ctx);                  \
        math::Copy(out_count, (int64_t*)data[1], index, ctx);             \
      } else {                                                            \
        _SelectViaDeviceSort<<<                                           \
            CUDA_BLOCKS(out_count),                                       \
            CUDA_THREADS,                                                 \
            0,                                                            \
            ctx->cuda_stream()>>>(                                        \
            out_count,                                                    \
            axis_dim,                                                     \
            inner_dim,                                                    \
            select_dim,                                                   \
            (T1*)data[0],                                                 \
            (int64_t*)data[1],                                            \
            value,                                                        \
            index);                                                       \
      }                                                                   \
      return;                                                             \
    }                                                                     \
    T2 init = largest > 0 ? kLowest : kMax;                               \
    DISPATCH_BLOCKSORT_KERNEL(T2);                                        \
  }

DEFINE_KERNEL_LAUNCHER(
    int8_t,
    int8_t,
    std::numeric_limits<int8_t>::lowest(),
    std::numeric_limits<int8_t>::max());
DEFINE_KERNEL_LAUNCHER(
    uint8_t,
    uint8_t,
    std::numeric_limits<uint8_t>::lowest(),
    std::numeric_limits<uint8_t>::max());
DEFINE_KERNEL_LAUNCHER(
    int,
    int,
    std::numeric_limits<int>::lowest(),
    std::numeric_limits<int>::max());
DEFINE_KERNEL_LAUNCHER(
    int64_t,
    int64_t,
    std::numeric_limits<int64_t>::lowest(),
    std::numeric_limits<int64_t>::max());
DEFINE_KERNEL_LAUNCHER(
    float16,
    half,
    cub::Traits<half>::Lowest(),
    cub::Traits<half>::Max());
DEFINE_KERNEL_LAUNCHER(
    float,
    float,
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::max());
DEFINE_KERNEL_LAUNCHER(
    double,
    double,
    std::numeric_limits<double>::lowest(),
    std::numeric_limits<double>::max());

#undef BLOCK_SORTKERNEL
#undef DISPATCH_BLOCKSORT_KERNEL
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
