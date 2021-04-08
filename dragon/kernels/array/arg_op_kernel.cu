#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, class CompareFunctor>
struct ArgFunctor {
  inline __device__ cub::KeyValuePair<int64_t, T> operator()(
      const cub::KeyValuePair<int64_t, T>& lhs,
      const cub::KeyValuePair<int64_t, T>& rhs) const {
    if ((compare_functor_(rhs.value, lhs.value)) ||
        (equal_functor_(lhs.value, rhs.value) && (rhs.key < lhs.key))) {
      return rhs;
    }
    return lhs;
  }
  CompareFunctor compare_functor_;
  math::EqualFunctor<T> equal_functor_;
};

template <typename T, class Reducer>
__global__ void _ArgReduce(
    const int NxS,
    const int S,
    const int C,
    const Reducer reducer,
    const T init,
    const T* x,
    int64_t* y) {
  typedef cub::KeyValuePair<int64_t, T> KeyValuePair;
  __shared__ typename BlockReduce<KeyValuePair>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    auto kv = KeyValuePair(-1, init);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      kv = reducer(kv, KeyValuePair(j, x[(i / S * C + j) * S + i % S]));
    }
    kv = BlockReduce<KeyValuePair>(storage).Reduce(kv, reducer);
    if (threadIdx.x == 0) {
      y[i] = kv.key;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T, CompareFunctor, kInit)                \
  template <>                                                                 \
  void name<T, CUDAContext>(                                                  \
      const int N,                                                            \
      const int S,                                                            \
      const int C,                                                            \
      const T* x,                                                             \
      int64_t* y,                                                             \
      CUDAContext* ctx) {                                                     \
    using ScalarT = math::ScalarType<T>::type;                                \
    const auto NxS = N * S;                                                   \
    _ArgReduce<<<CUDA_2D_BLOCKS(NxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        NxS,                                                                  \
        S,                                                                    \
        C,                                                                    \
        ArgFunctor<ScalarT, CompareFunctor<ScalarT>>(),                       \
        kInit,                                                                \
        reinterpret_cast<const ScalarT*>(x),                                  \
        y);                                                                   \
  }

DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    uint8_t,
    math::GreaterFunctor,
    std::numeric_limits<uint8_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    int8_t,
    math::GreaterFunctor,
    std::numeric_limits<int8_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    int,
    math::GreaterFunctor,
    std::numeric_limits<int>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    int64_t,
    math::GreaterFunctor,
    std::numeric_limits<int64_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    float16,
    math::GreaterFunctor,
    cub::Traits<half>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    float,
    math::GreaterFunctor,
    std::numeric_limits<float>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    double,
    math::GreaterFunctor,
    std::numeric_limits<double>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    uint8_t,
    math::LessFunctor,
    std::numeric_limits<uint8_t>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    int8_t,
    math::LessFunctor,
    std::numeric_limits<int8_t>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    int,
    math::LessFunctor,
    std::numeric_limits<int>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    int64_t,
    math::LessFunctor,
    std::numeric_limits<int64_t>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    float16,
    math::LessFunctor,
    cub::Traits<half>::Max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    float,
    math::LessFunctor,
    std::numeric_limits<float>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    double,
    math::LessFunctor,
    std::numeric_limits<double>::max());
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
