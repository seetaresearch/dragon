#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
struct ArgMaxFunctor {
  inline __device__ cub::KeyValuePair<int64_t, T> operator()(
      const cub::KeyValuePair<int64_t, T>& lhs,
      const cub::KeyValuePair<int64_t, T>& rhs) const {
    if ((greater_(rhs.value, lhs.value)) ||
        (equal_(lhs.value, rhs.value) && (rhs.key < lhs.key))) {
      return rhs;
    }
    return lhs;
  }
  math::GreaterFunctor<T> greater_;
  math::EqualFunctor<T> equal_;
};

template <typename T>
struct ArgMinFunctor {
  inline __device__ cub::KeyValuePair<int64_t, T> operator()(
      const cub::KeyValuePair<int64_t, T>& lhs,
      const cub::KeyValuePair<int64_t, T>& rhs) const {
    if ((less_(rhs.value, lhs.value)) ||
        (equal_(lhs.value, rhs.value) && (rhs.key < lhs.key))) {
      return rhs;
    }
    return lhs;
  }
  math::LessFunctor<T> less_;
  math::EqualFunctor<T> equal_;
};

template <typename T, class Reducer>
__global__ void _ArgReduce(
    const int rows,
    const int cols,
    const int inner_dim,
    const Reducer reducer,
    const T init,
    const T* x,
    int64_t* y) {
  typedef cub::KeyValuePair<int64_t, T> KeyValuePair;
  __shared__ typename BlockReduce<KeyValuePair>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    auto key_val = KeyValuePair(-1, init);
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      key_val = reducer(
          key_val,
          KeyValuePair(
              j, x[((i / inner_dim) * cols + j) * inner_dim + i % inner_dim]));
    }
    key_val = BlockReduce<KeyValuePair>(storage).Reduce(key_val, reducer);
    if (threadIdx.x == 0) {
      y[i] = key_val.key;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T1, T2, Reducer, kInit)                   \
  template <>                                                                  \
  void name<T1, CUDAContext>(                                                  \
      const int outer_dim,                                                     \
      const int inner_dim,                                                     \
      const int axis_dim,                                                      \
      const T1* x,                                                             \
      int64_t* y,                                                              \
      CUDAContext* ctx) {                                                      \
    const auto rows = outer_dim * inner_dim;                                   \
    const auto cols = axis_dim;                                                \
    _ArgReduce<<<CUDA_2D_BLOCKS(rows), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        rows,                                                                  \
        cols,                                                                  \
        inner_dim,                                                             \
        Reducer<T2>(),                                                         \
        kInit,                                                                 \
        reinterpret_cast<const T2*>(x),                                        \
        y);                                                                    \
  }

DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    int8_t,
    int8_t,
    ArgMaxFunctor,
    std::numeric_limits<int8_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    uint8_t,
    uint8_t,
    ArgMaxFunctor,
    std::numeric_limits<uint8_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    int,
    int,
    ArgMaxFunctor,
    std::numeric_limits<int>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    int64_t,
    int64_t,
    ArgMaxFunctor,
    std::numeric_limits<int64_t>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    float16,
    half,
    ArgMaxFunctor,
    cub::Traits<half>::Lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    float,
    float,
    ArgMaxFunctor,
    std::numeric_limits<float>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMax,
    double,
    double,
    ArgMaxFunctor,
    std::numeric_limits<double>::lowest());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    int8_t,
    int8_t,
    ArgMinFunctor,
    std::numeric_limits<int8_t>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    uint8_t,
    uint8_t,
    ArgMinFunctor,
    std::numeric_limits<uint8_t>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    int,
    int,
    ArgMinFunctor,
    std::numeric_limits<int>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    int64_t,
    int64_t,
    ArgMinFunctor,
    std::numeric_limits<int64_t>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    float16,
    half,
    ArgMinFunctor,
    cub::Traits<half>::Max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    float,
    float,
    ArgMinFunctor,
    std::numeric_limits<float>::max());
DEFINE_KERNEL_LAUNCHER(
    ArgMin,
    double,
    double,
    ArgMinFunctor,
    std::numeric_limits<double>::max());
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
