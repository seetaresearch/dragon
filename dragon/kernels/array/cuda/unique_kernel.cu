#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

__global__ void _RemapInverse(
    const int dim,
    const int num,
    thrust::device_ptr<int> order1,
    thrust::device_ptr<int> order2,
    int64_t* inverse_index) {
  const int yi = blockDim.x * blockIdx.x + threadIdx.x;
  if (yi >= num) return;
  int xi = order2[yi];
  inverse_index[order1[xi]] = yi;
  for (xi++; xi < dim && (yi == num - 1 || xi != order2[yi + 1]); xi++) {
    inverse_index[order1[xi]] = yi;
  }
}

__global__ void _ComputeCounts(
    const int dim,
    const int num,
    thrust::device_ptr<int> order2,
    int64_t* counts) {
  const int yi = blockDim.x * blockIdx.x + threadIdx.x;
  if (yi >= num) return;
  counts[yi] = (yi == num - 1 ? dim : order2[yi + 1]) - order2[yi];
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void Unique<T, CUDAContext>(                                                 \
      const int dim,                                                           \
      const T* x,                                                              \
      T* y,                                                                    \
      int64_t* inverse_index,                                                  \
      int64_t* counts,                                                         \
      int* num,                                                                \
      CUDAContext* ctx) {                                                      \
    math::Copy(dim, x, y, ctx);                                                \
    auto policy = thrust::cuda::par.on(ctx->cuda_stream());                    \
    auto* data = reinterpret_cast<math::ScalarType<T>::type*>(y);              \
    thrust::device_vector<int> order1(dim), order2(dim);                       \
    thrust::sequence(policy, order1.begin(), order1.end());                    \
    thrust::sequence(policy, order2.begin(), order2.end());                    \
    thrust::sort_by_key(                                                       \
        policy,                                                                \
        data,                                                                  \
        data + dim,                                                            \
        order1.begin(),                                                        \
        math::LessFunctor<math::ScalarType<T>::type>());                       \
    auto last = thrust::unique_by_key(                                         \
        policy,                                                                \
        data,                                                                  \
        data + dim,                                                            \
        order2.begin(),                                                        \
        math::EqualFunctor<math::ScalarType<T>::type>());                      \
    int n = num[0] = last.first - data;                                        \
    if (inverse_index) {                                                       \
      _RemapInverse<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
          dim, n, order1.data(), order2.data(), inverse_index);                \
    }                                                                          \
    if (counts) {                                                              \
      _ComputeCounts<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          dim, n, order2.data(), counts);                                      \
    }                                                                          \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
