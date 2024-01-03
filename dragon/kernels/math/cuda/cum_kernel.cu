#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, class Reducer>
struct BlockPrefixOp {
  inline __device__ BlockPrefixOp(const T init) : global_aggr_(init) {}
  inline __device__ T operator()(const T block_aggr) {
    const T current_aggr = global_aggr_;
    global_aggr_ = reducer_(global_aggr_, block_aggr);
    return current_aggr;
  }
  T global_aggr_;
  Reducer reducer_;
};

template <typename T, typename AccT, class Reducer>
__global__ void _CumReduce(
    const int NxS,
    const int S,
    const int C,
    const Reducer reducer,
    const AccT init,
    const bool exclusive,
    const bool reverse,
    const T* x,
    T* y) {
  __shared__ typename BlockScan<AccT>::TempStorage storage;
  const int C_rounded = math::utils::DivUp<int>(C, blockDim.x) * blockDim.x;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    const int row_offset = i / S * C * S + i % S;
    BlockPrefixOp<AccT, Reducer> prefix_op(init);
    for (int j = threadIdx.x; j < C_rounded; j += blockDim.x) {
      const int index = row_offset + (reverse ? (C - j - 1) : j) * S;
      AccT val = j < C ? AccT(x[index]) : init;
      if (exclusive) {
        BlockScan<AccT>(storage).ExclusiveScan(val, val, reducer, prefix_op);
      } else {
        BlockScan<AccT>(storage).InclusiveScan(val, val, reducer, prefix_op);
      }
      if (j < C) y[index] = val;
      __syncthreads();
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, Reducer, kInit)               \
  template <>                                                         \
  void name<T, CUDAContext>(                                          \
      const int N,                                                    \
      const int S,                                                    \
      const int C,                                                    \
      const bool exclusive,                                           \
      const bool reverse,                                             \
      const T* x,                                                     \
      T* y,                                                           \
      CUDAContext* ctx) {                                             \
    using AccT = math::Traits<T>::accumulator_type;                   \
    const auto NxS = N * S;                                           \
    _CumReduce<math::Traits<T>::scalar_type, AccT, Reducer<AccT>>     \
        <<<CUDA_BLOCKS(NxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
            NxS,                                                      \
            S,                                                        \
            C,                                                        \
            Reducer<AccT>(),                                          \
            kInit,                                                    \
            exclusive,                                                \
            reverse,                                                  \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x), \
            reinterpret_cast<math::Traits<T>::scalar_type*>(y));      \
  }

// clang-format off
DEFINE_KERNEL_LAUNCHER(CumSum, uint8_t, math::PlusFunctor, uint8_t(0));
DEFINE_KERNEL_LAUNCHER(CumSum, int8_t, math::PlusFunctor, int8_t(0));
DEFINE_KERNEL_LAUNCHER(CumSum, int, math::PlusFunctor, int(0));
DEFINE_KERNEL_LAUNCHER(CumSum, int64_t, math::PlusFunctor, int64_t(0));
DEFINE_KERNEL_LAUNCHER(CumSum, float16, math::PlusFunctor, 0.f);
DEFINE_KERNEL_LAUNCHER(CumSum, bfloat16, math::PlusFunctor, 0.f);
DEFINE_KERNEL_LAUNCHER(CumSum, float, math::PlusFunctor, 0.f);
DEFINE_KERNEL_LAUNCHER(CumSum, double, math::PlusFunctor, 0.);
DEFINE_KERNEL_LAUNCHER(CumMax, uint8_t, math::MaxFunctor, math::Traits<uint8_t>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, int8_t, math::MaxFunctor, math::Traits<int8_t>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, int, math::MaxFunctor, math::Traits<int>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, int64_t, math::MaxFunctor, math::Traits<int64_t>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, float16, math::MaxFunctor, math::Traits<float16>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, bfloat16, math::MaxFunctor, math::Traits<bfloat16>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, float, math::MaxFunctor, math::Traits<float>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, double, math::MaxFunctor, math::Traits<double>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMin, uint8_t, math::MinFunctor, math::Traits<uint8_t>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, int8_t, math::MinFunctor, math::Traits<int8_t>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, int, math::MinFunctor, math::Traits<int>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, int64_t, math::MinFunctor, math::Traits<int64_t>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, float16, math::MinFunctor, math::Traits<float16>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, bfloat16, math::MinFunctor, math::Traits<bfloat16>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, float, math::MinFunctor, math::Traits<float>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, double, math::MinFunctor, math::Traits<double>::Max());
#undef DEFINE_KERNEL_LAUNCHER // clang-format on

} // namespace kernels

} // namespace dragon
