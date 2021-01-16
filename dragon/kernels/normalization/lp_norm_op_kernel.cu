#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T, typename AccT>
__global__ void _L1Normalize(
    const int nblocks,
    const int inner_dim,
    const int reduce_dim,
    const AccT normalizer,
    const AccT epsilon,
    const T* x,
    T* y) {
  __shared__ AccT norm;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    AccT sum = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      sum += abs(convert::To<AccT>(x[offset + j * inner_dim]));
    }
    sum = BlockReduce<AccT>(storage).Sum(sum);
    if (threadIdx.x == 0) {
      norm = max(sum / normalizer, epsilon);
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      y[idx] = convert::To<T>(convert::To<AccT>(x[idx]) / norm);
    }
  }
}

template <typename T, typename AccT>
__global__ void _L2Normalize(
    const int nblocks,
    const int inner_dim,
    const int reduce_dim,
    const AccT normalizer,
    const AccT epsilon,
    const T* x,
    T* y) {
  __shared__ AccT norm;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    AccT sum = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      sum += math::utils::Square(convert::To<AccT>(x[offset + j * inner_dim]));
    }
    sum = BlockReduce<AccT>(storage).Sum(sum);
    if (threadIdx.x == 0) {
      norm = max(sqrt(sum / normalizer), epsilon);
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      y[idx] = convert::To<T>(convert::To<AccT>(x[idx]) / norm);
    }
  }
}

template <typename T, typename AccT>
__global__ void _L1NormalizeGrad(
    const int nblocks,
    const int inner_dim,
    const int reduce_dim,
    const AccT normalizer,
    const AccT epsilon,
    const T* dy,
    const T* x,
    T* dx) {
  __shared__ AccT norm, norm2, sum;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    AccT val1 = AccT(0), val2 = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      val1 += abs(convert::To<AccT>(x[idx]));
      val2 += convert::To<AccT>(dy[idx]) * convert::To<AccT>(x[idx]);
    }
    val1 = BlockReduce<AccT>(storage).Sum(val1);
    val2 = BlockReduce<AccT>(storage).Sum(val2);
    if (threadIdx.x == 0) {
      norm = max(val1 / normalizer, epsilon);
      norm2 = pow(norm, AccT(2));
      sum = val2 / normalizer;
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      dx[idx] = convert::To<T>(
          (convert::To<AccT>(dy[idx]) / norm) -
          ((math::utils::Sign(convert::To<AccT>(x[idx])) / norm2) * sum));
    }
  }
}

template <typename T, typename AccT>
__global__ void _L2NormalizeGrad(
    const int nblocks,
    const int inner_dim,
    const int reduce_dim,
    const AccT normalizer,
    const AccT epsilon,
    const T* dy,
    const T* x,
    T* dx) {
  __shared__ AccT norm, norm3, sum;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    AccT val1 = AccT(0), val2 = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      val1 += math::utils::Square(convert::To<AccT>(x[idx]));
      val2 += convert::To<AccT>(dy[idx]) * convert::To<AccT>(x[idx]);
    }
    val1 = BlockReduce<AccT>(storage).Sum(val1);
    val2 = BlockReduce<AccT>(storage).Sum(val2);
    if (threadIdx.x == 0) {
      norm = max(sqrt(val1 / normalizer), epsilon);
      norm3 = pow(norm, AccT(3));
      sum = val2 / normalizer;
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      dx[idx] = convert::To<T>(
          (convert::To<AccT>(dy[idx]) / norm) -
          ((convert::To<AccT>(x[idx]) / norm3) * sum));
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T, AccT)                               \
  template <>                                                               \
  void name<T, CUDAContext>(                                                \
      const int outer_dim,                                                  \
      const int inner_dim,                                                  \
      const int reduce_dim,                                                 \
      const float normalizer,                                               \
      const float epsilon,                                                  \
      const T* x,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    const auto nblocks = outer_dim * inner_dim;                             \
    _##name<math::ScalarType<T>::type, AccT>                                \
        <<<CUDA_2D_BLOCKS(nblocks), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
            nblocks,                                                        \
            inner_dim,                                                      \
            reduce_dim,                                                     \
            AccT(normalizer),                                               \
            AccT(epsilon),                                                  \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),          \
            reinterpret_cast<math::ScalarType<T>::type*>(y));               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T, AccT)                          \
  template <>                                                               \
  void name<T, CUDAContext>(                                                \
      const int outer_dim,                                                  \
      const int inner_dim,                                                  \
      const int reduce_dim,                                                 \
      const float normalizer,                                               \
      const float epsilon,                                                  \
      const T* dy,                                                          \
      const T* x,                                                           \
      T* dx,                                                                \
      CUDAContext* ctx) {                                                   \
    const auto nblocks = outer_dim * inner_dim;                             \
    _##name<math::ScalarType<T>::type, AccT>                                \
        <<<CUDA_2D_BLOCKS(nblocks), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
            nblocks,                                                        \
            inner_dim,                                                      \
            reduce_dim,                                                     \
            AccT(normalizer),                                               \
            AccT(epsilon),                                                  \
            reinterpret_cast<const math::ScalarType<T>::type*>(dy),         \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),          \
            reinterpret_cast<math::ScalarType<T>::type*>(dx));              \
  }

DEFINE_KERNEL_LAUNCHER(L1Normalize, float16, float);
DEFINE_KERNEL_LAUNCHER(L1Normalize, float, float);
DEFINE_KERNEL_LAUNCHER(L1Normalize, double, double);
DEFINE_KERNEL_LAUNCHER(L2Normalize, float16, float);
DEFINE_KERNEL_LAUNCHER(L2Normalize, float, float);
DEFINE_KERNEL_LAUNCHER(L2Normalize, double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormalizeGrad, float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormalizeGrad, float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormalizeGrad, double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormalizeGrad, float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormalizeGrad, float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormalizeGrad, double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
