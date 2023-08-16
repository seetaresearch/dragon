#include "dragon/core/workspace.h"
#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, class CompareFunctor>
struct SortFunctor {
  SortFunctor(int64_t CxS, int64_t S) : CxS_(CxS), S_(S) {}
  inline __device__ bool operator()(
      const thrust::tuple<int64_t, T>& lhs,
      const thrust::tuple<int64_t, T>& rhs) const {
    int64_t i = thrust::get<0>(lhs), j = thrust::get<0>(rhs);
    i = (i / CxS_) * S_ + i % S_;
    j = (j / CxS_) * S_ + j % S_;
    if (i != j) {
      return i < j;
    } else {
      return compare_functor_(thrust::get<1>(lhs), thrust::get<1>(rhs));
    }
  }
  int64_t CxS_, S_;
  CompareFunctor compare_functor_;
};

template <typename T, int ItemsPerThread>
__global__ void _BlockSort(
    const int NxS,
    const int S,
    const int C,
    const int K,
    const bool largest,
    const T init,
    const T* x,
    T* y,
    int64_t* index) {
  typedef cub::BlockRadixSort<T, CUDA_THREADS, ItemsPerThread, int64_t>
      BlockSort;
  __shared__ typename BlockSort::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    T keys[ItemsPerThread];
    int64_t values[ItemsPerThread];
    const int thread_offset = threadIdx.x * ItemsPerThread;
    const int x_offset = i / S * C * S + i % S;
    const int y_offset = i / S * K * S + i % S;
#pragma unroll
    for (int j = 0; j < ItemsPerThread; ++j) {
      const int item_idx = thread_offset + j;
      values[j] = item_idx < C ? item_idx : C - 1;
      keys[j] = item_idx < C ? x[x_offset + item_idx * S] : init;
    }
    __syncthreads();
    if (largest) {
      BlockSort(storage).SortDescending(keys, values);
    } else {
      BlockSort(storage).Sort(keys, values);
    }
#pragma unroll
    for (int j = 0; j < ItemsPerThread; ++j) {
      if (thread_offset + j < K) {
        y[y_offset + (thread_offset + j) * S] = keys[j];
        index[y_offset + (thread_offset + j) * S] = values[j];
      }
    }
  }
}

template <typename T>
void _DeviceSort(
    const int N,
    const int S,
    const int C,
    const int largest,
    T* key,
    int64_t* value,
    CUDAContext* ctx) {
  const auto NxS = N * S;
  const auto NxSxC = NxS * C;
  auto policy = thrust::cuda::par.on(ctx->cuda_stream());
  thrust::sequence(policy, value, value + NxSxC);
  if (NxS == 1) {
    if (largest > 0) {
      thrust::sort_by_key(
          policy, key, key + NxSxC, value, math::GreaterFunctor<T>());
    } else {
      thrust::sort_by_key(
          policy, key, key + NxSxC, value, math::LessFunctor<T>());
    }
  } else {
    auto kv = thrust::make_zip_iterator(thrust::make_tuple(value, key));
    if (largest > 0) {
      thrust::sort(
          policy,
          kv,
          kv + NxSxC,
          SortFunctor<T, math::GreaterFunctor<T>>(C * S, S));
    } else {
      thrust::sort(
          policy,
          kv,
          kv + NxSxC,
          SortFunctor<T, math::LessFunctor<T>>(C * S, S));
    }
  }
}

template <typename T>
__global__ void _GetTopK(
    const int NxKxS,
    const int S,
    const int C,
    const int K,
    const T* key,
    const int64_t* value,
    T* y,
    int64_t* index) {
  CUDA_1D_KERNEL_LOOP(yi, NxKxS) {
    const int xi = ((yi / S / K) * S + yi % S) * C + (yi / S) % K;
    y[yi] = key[xi];
    index[yi] = value[xi] / S % C;
  }
}

} // namespace

#define DISPATCH_BLOCKSORT_KERNEL(T, kItemsPerThread) \
  _BlockSort<T, kItemsPerThread>                      \
      <<<NxS, CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          NxS,                                        \
          S,                                          \
          C,                                          \
          K,                                          \
          largest > 0,                                \
          init,                                       \
          reinterpret_cast<const T*>(x),              \
          reinterpret_cast<T*>(value),                \
          index)

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void TopK<T, CUDAContext>(                                                   \
      const int N,                                                             \
      const int S,                                                             \
      const int C,                                                             \
      const int K,                                                             \
      const int largest,                                                       \
      const T* x,                                                              \
      T* value,                                                                \
      int64_t* index,                                                          \
      CUDAContext* ctx) {                                                      \
    const int NxS = N * S;                                                     \
    if (NxS == 1 || C > CUDA_THREADS * 8) {                                    \
      const auto NxCxS = N * C * S;                                            \
      const auto NxKxS = N * K * S;                                            \
      auto* scratch = ctx->workspace()->template data<CUDAContext>(            \
          NxCxS * (sizeof(T) + sizeof(int64_t)), "BufferKernel");              \
      auto* scratch2 = (int64_t*)((uint8_t*)scratch + NxCxS * sizeof(T));      \
      math::Copy(NxCxS, x, (T*)scratch, ctx);                                  \
      _DeviceSort(                                                             \
          N,                                                                   \
          S,                                                                   \
          C,                                                                   \
          largest,                                                             \
          reinterpret_cast<math::Traits<T>::scalar_type*>(scratch),            \
          scratch2,                                                            \
          ctx);                                                                \
      if (NxS == 1) {                                                          \
        math::Copy(NxKxS, (T*)scratch, value, ctx);                            \
        math::Copy(NxKxS, scratch2, index, ctx);                               \
      } else {                                                                 \
        _GetTopK<<<CUDA_BLOCKS(NxKxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
            NxKxS, S, C, K, (T*)scratch, scratch2, value, index);              \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
    auto init = largest ? math::Traits<T>::Lowest() : math::Traits<T>::Max();  \
    if (C <= CUDA_THREADS) {                                                   \
      DISPATCH_BLOCKSORT_KERNEL(math::Traits<T>::scalar_type, 1);              \
    } else if (C <= CUDA_THREADS * 2) {                                        \
      DISPATCH_BLOCKSORT_KERNEL(math::Traits<T>::scalar_type, 2);              \
    } else if (C <= CUDA_THREADS * 4) {                                        \
      DISPATCH_BLOCKSORT_KERNEL(math::Traits<T>::scalar_type, 4);              \
    } else if (C <= CUDA_THREADS * 8) {                                        \
      DISPATCH_BLOCKSORT_KERNEL(math::Traits<T>::scalar_type, 8);              \
    } else {                                                                   \
      LOG(FATAL) << "Too larger dimension (> " << CUDA_THREADS * 8             \
                 << ") to launch the cuda kernel";                             \
    }                                                                          \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DISPATCH_BLOCKSORT_KERNEL
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
