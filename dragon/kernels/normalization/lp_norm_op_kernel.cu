#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _L1Norm(
    const int NxS,
    const int S,
    const int C,
    const AccT normalizer,
    const AccT epsilon,
    const T* x,
    T* y) {
  __shared__ AccT norm;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    auto offset = i / S * C * S + i % S;
    AccT sum = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      sum += abs(convert::To<AccT>(x[offset + j * S]));
    }
    sum = BlockReduce<AccT>(storage).Sum(sum);
    if (threadIdx.x == 0) {
      norm = max(sum / normalizer, epsilon);
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, C) {
      auto index = offset + j * S;
      y[index] = convert::To<T>(convert::To<AccT>(x[index]) / norm);
    }
  }
}

template <typename T, typename AccT>
__global__ void _L2Norm(
    const int NxS,
    const int S,
    const int C,
    const AccT normalizer,
    const AccT epsilon,
    const T* x,
    T* y) {
  __shared__ AccT norm;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    auto offset = i / S * C * S + i % S;
    AccT sum = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      sum += math::utils::Square(convert::To<AccT>(x[offset + j * S]));
    }
    sum = BlockReduce<AccT>(storage).Sum(sum);
    if (threadIdx.x == 0) {
      norm = max(sqrt(sum / normalizer), epsilon);
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, C) {
      auto index = offset + j * S;
      y[index] = convert::To<T>(convert::To<AccT>(x[index]) / norm);
    }
  }
}

template <typename T, typename AccT>
__global__ void _L1NormGrad(
    const int NxS,
    const int S,
    const int C,
    const AccT normalizer,
    const AccT epsilon,
    const T* dy,
    const T* x,
    T* dx) {
  __shared__ AccT norm, norm2, sum;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    auto offset = i / S * C * S + i % S;
    AccT val1 = AccT(0), val2 = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      auto index = offset + j * S;
      val1 += abs(convert::To<AccT>(x[index]));
      val2 += convert::To<AccT>(dy[index]) * convert::To<AccT>(x[index]);
    }
    val1 = BlockReduce<AccT>(storage).Sum(val1);
    val2 = BlockReduce<AccT>(storage).Sum(val2);
    if (threadIdx.x == 0) {
      norm = max(val1 / normalizer, epsilon);
      norm2 = pow(norm, AccT(2));
      sum = val2 / normalizer;
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, C) {
      auto index = offset + j * S;
      dx[index] = convert::To<T>(
          (convert::To<AccT>(dy[index]) / norm) -
          ((math::utils::Sign(convert::To<AccT>(x[index])) / norm2) * sum));
    }
  }
}

template <typename T, typename AccT>
__global__ void _L2NormGrad(
    const int NxS,
    const int S,
    const int C,
    const AccT normalizer,
    const AccT epsilon,
    const T* dy,
    const T* x,
    T* dx) {
  __shared__ AccT norm, norm3, sum;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    auto offset = i / S * C * S + i % S;
    AccT val1 = AccT(0), val2 = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      auto index = offset + j * S;
      val1 += math::utils::Square(convert::To<AccT>(x[index]));
      val2 += convert::To<AccT>(dy[index]) * convert::To<AccT>(x[index]);
    }
    val1 = BlockReduce<AccT>(storage).Sum(val1);
    val2 = BlockReduce<AccT>(storage).Sum(val2);
    if (threadIdx.x == 0) {
      norm = max(sqrt(val1 / normalizer), epsilon);
      norm3 = pow(norm, AccT(3));
      sum = val2 / normalizer;
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, C) {
      auto index = offset + j * S;
      dx[index] = convert::To<T>(
          (convert::To<AccT>(dy[index]) / norm) -
          ((convert::To<AccT>(x[index]) / norm3) * sum));
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T, AccT)                      \
  template <>                                                      \
  void name<T, CUDAContext>(                                       \
      const int N,                                                 \
      const int S,                                                 \
      const int C,                                                 \
      const float normalizer,                                      \
      const float epsilon,                                         \
      const T* x,                                                  \
      T* y,                                                        \
      CUDAContext* ctx) {                                          \
    const auto NxS = N * S;                                        \
    _##name<math::ScalarType<T>::type, AccT>                       \
        <<<NxS, CUDA_THREADS, 0, ctx->cuda_stream()>>>(            \
            NxS,                                                   \
            S,                                                     \
            C,                                                     \
            AccT(normalizer),                                      \
            AccT(epsilon),                                         \
            reinterpret_cast<const math::ScalarType<T>::type*>(x), \
            reinterpret_cast<math::ScalarType<T>::type*>(y));      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T, AccT)                  \
  template <>                                                       \
  void name<T, CUDAContext>(                                        \
      const int N,                                                  \
      const int S,                                                  \
      const int C,                                                  \
      const float normalizer,                                       \
      const float epsilon,                                          \
      const T* dy,                                                  \
      const T* x,                                                   \
      T* dx,                                                        \
      CUDAContext* ctx) {                                           \
    const auto NxS = N * S;                                         \
    _##name<math::ScalarType<T>::type, AccT>                        \
        <<<NxS, CUDA_THREADS, 0, ctx->cuda_stream()>>>(             \
            NxS,                                                    \
            S,                                                      \
            C,                                                      \
            AccT(normalizer),                                       \
            AccT(epsilon),                                          \
            reinterpret_cast<const math::ScalarType<T>::type*>(dy), \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),  \
            reinterpret_cast<math::ScalarType<T>::type*>(dx));      \
  }

DEFINE_KERNEL_LAUNCHER(L1Norm, float16, float);
DEFINE_KERNEL_LAUNCHER(L1Norm, float, float);
DEFINE_KERNEL_LAUNCHER(L1Norm, double, double);
DEFINE_KERNEL_LAUNCHER(L2Norm, float16, float);
DEFINE_KERNEL_LAUNCHER(L2Norm, float, float);
DEFINE_KERNEL_LAUNCHER(L2Norm, double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
