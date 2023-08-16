#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"

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

#define DEFINE_KERNEL_LAUNCHER(name, T, CompareFunctor, kInit) \
  template <>                                                  \
  void name<T, CUDAContext>(                                   \
      const int N,                                             \
      const int S,                                             \
      const int C,                                             \
      const T* x,                                              \
      int64_t* y,                                              \
      CUDAContext* ctx) {                                      \
    using ScalarT = math::Traits<T>::scalar_type;              \
    const auto NxS = N * S;                                    \
    _ArgReduce<<<NxS, CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
        NxS,                                                   \
        S,                                                     \
        C,                                                     \
        ArgFunctor<ScalarT, CompareFunctor<ScalarT>>(),        \
        convert::To<ScalarT>(kInit),                           \
        reinterpret_cast<const ScalarT*>(x),                   \
        y);                                                    \
  }

DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    uint8_t,
    math::GreaterFunctor,
    math::Traits<uint8_t>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    int8_t,
    math::GreaterFunctor,
    math::Traits<int8_t>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    int,
    math::GreaterFunctor,
    math::Traits<int>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    int64_t,
    math::GreaterFunctor,
    math::Traits<int64_t>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    float16,
    math::GreaterFunctor,
    math::Traits<float16>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    bfloat16,
    math::GreaterFunctor,
    math::Traits<bfloat16>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    float,
    math::GreaterFunctor,
    math::Traits<float>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    double,
    math::GreaterFunctor,
    math::Traits<double>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    uint8_t,
    math::LessFunctor,
    math::Traits<uint8_t>::Max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    int8_t,
    math::LessFunctor,
    math::Traits<int8_t>::Max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    int,
    math::LessFunctor,
    math::Traits<int>::Max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    int64_t,
    math::LessFunctor,
    math::Traits<int64_t>::Max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    float16,
    math::LessFunctor,
    math::Traits<float16>::Max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    bfloat16,
    math::LessFunctor,
    math::Traits<bfloat16>::Max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    float,
    math::LessFunctor,
    math::Traits<float>::Max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    double,
    math::LessFunctor,
    math::Traits<double>::Max());
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
