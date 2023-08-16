#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _Sequence(const int N, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = convert::To<T>(float(i));
  }
}

template <typename T>
__global__ void
_Range(const int N, const double start, const double delta, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = convert::To<T>(start + double(i) * delta);
  }
}

template <typename T, int D>
__global__ void _RowwiseLinSpace(
    const int N,
    const int C,
    const SimpleArray<double, D> Y_starts,
    const SimpleArray<double, D> Y_stops,
    T* y) {
  const auto NxC = N * C;
  CUDA_1D_KERNEL_LOOP(index, NxC) {
    const int i = index % C;
    const int j = index / C;
    if (j == N - 1 && j > 0) {
      y[index] = convert::To<T>(Y_stops.data[i]);
    } else {
      y[index] = convert::To<T>(
          Y_starts.data[i] +
          j * ((Y_stops.data[i] - Y_starts.data[i]) / double(N - 1)));
    }
  }
}

template <typename T, int D>
__global__ void _ColwiseLinSpace(
    const int NxC,
    const int C,
    const SimpleArray<double, D> Y_starts,
    const SimpleArray<double, D> Y_stops,
    T* y) {
  CUDA_1D_KERNEL_LOOP(index, NxC) {
    const int i = index / C;
    const int j = index % C;
    if (j == C - 1 && j > 0) {
      y[index] = convert::To<T>(Y_stops.data[i]);
    } else {
      y[index] = convert::To<T>(
          Y_starts.data[i] +
          j * ((Y_stops.data[i] - Y_starts.data[i]) / double(C - 1)));
    }
  }
}

template <typename T, bool kUpper>
__global__ void
_SetEye(const int nthreads, const int M, const int N, const int k, T* y) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int i = index % M;
    if (kUpper) {
      const int j = i + k;
      y[index * N + min(j, N - 1)] =
          j < N ? convert::To<T>(1.f) : convert::To<T>(0.f);
    } else {
      const int j = i - k;
      y[index * N + max(j, 0)] =
          j < 0 ? convert::To<T>(0.f) : convert::To<T>(1.f);
    }
  }
}

template <typename T, bool kUpper>
__global__ void
_SetTrilu(const int nthreads, const int M, const int N, const int k, T* y) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int j = index % N;
    const int i = (index / N) % M;
    if (kUpper) {
      y[index] = j < i + k ? convert::To<T>(0.f) : y[index];
    } else {
      y[index] = j > i + k ? convert::To<T>(0.f) : y[index];
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void Range<T, CUDAContext>(                                                 \
      const int N,                                                            \
      const double start,                                                     \
      const double delta,                                                     \
      T* y,                                                                   \
      CUDAContext* ctx) {                                                     \
    _Range<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(          \
        N, start, delta, reinterpret_cast<math::Traits<T>::scalar_type*>(y)); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                              \
  template <>                                                  \
  void LinSpace<T, CUDAContext>(                               \
      const int N,                                             \
      const int C,                                             \
      const int axis,                                          \
      const double* starts,                                    \
      const double* stops,                                     \
      T* y,                                                    \
      CUDAContext* ctx) {                                      \
    CUDA_TENSOR_DIMS_CHECK((axis == 0 ? C : N));               \
    const auto NxC = N * C;                                    \
    SimpleArray<double, CUDA_TENSOR_MAX_DIMS> Y_starts;        \
    SimpleArray<double, CUDA_TENSOR_MAX_DIMS> Y_stops;         \
    for (int i = 0; i < (axis == 0 ? C : N); ++i) {            \
      Y_starts.data[i] = starts[i];                            \
      Y_stops.data[i] = stops[i];                              \
    }                                                          \
    if (axis == 0) {                                           \
      _RowwiseLinSpace<<<                                      \
          CUDA_BLOCKS(NxC),                                    \
          CUDA_THREADS,                                        \
          0,                                                   \
          ctx->cuda_stream()>>>(                               \
          N,                                                   \
          C,                                                   \
          Y_starts,                                            \
          Y_stops,                                             \
          reinterpret_cast<math::Traits<T>::scalar_type*>(y)); \
    } else {                                                   \
      _ColwiseLinSpace<<<                                      \
          CUDA_BLOCKS(NxC),                                    \
          CUDA_THREADS,                                        \
          0,                                                   \
          ctx->cuda_stream()>>>(                               \
          NxC,                                                 \
          C,                                                   \
          Y_starts,                                            \
          Y_stops,                                             \
          reinterpret_cast<math::Traits<T>::scalar_type*>(y)); \
    }                                                          \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                       \
  template <>                                                           \
  void Permutation<T, CUDAContext>(                                     \
      const int N, const uint32_t* r, T* y, CUDAContext* ctx) {         \
    auto keys = thrust::device_ptr<uint32_t>(const_cast<uint32_t*>(r)); \
    auto values = thrust::device_ptr<T>(y);                             \
    auto policy = thrust::cuda::par.on(ctx->cuda_stream());             \
    thrust::sequence(policy, values, values + N);                       \
    thrust::sort_by_key(policy, keys, keys + N, values);                \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void Permutation<T, CUDAContext>(                                           \
      const int N, const uint32_t* r, T* y, CUDAContext* ctx) {               \
    using ScalarT = math::Traits<T>::scalar_type;                             \
    auto keys = thrust::device_ptr<uint32_t>(const_cast<uint32_t*>(r));       \
    auto values = thrust::device_ptr<ScalarT>(reinterpret_cast<ScalarT*>(y)); \
    auto policy = thrust::cuda::par.on(ctx->cuda_stream());                   \
    _Sequence<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(       \
        N, reinterpret_cast<ScalarT*>(y));                                    \
    thrust::sort_by_key(policy, keys, keys + N, values);                      \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void SetEye<T, CUDAContext>(                                              \
      const int batch_size,                                                 \
      const int M,                                                          \
      const int N,                                                          \
      const int k,                                                          \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    const auto nthreads = batch_size * M;                                   \
    math::Set((nthreads * N), convert::To<T>(0.f), y, ctx);                 \
    if (k > 0) {                                                            \
      _SetEye<math::Traits<T>::scalar_type, true>                           \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              nthreads, M, N, k, (math::Traits<T>::scalar_type*)y);         \
    } else {                                                                \
      _SetEye<math::Traits<T>::scalar_type, false>                          \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              nthreads, M, N, -k, (math::Traits<T>::scalar_type*)y);        \
    }                                                                       \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void SetTrilu<T, CUDAContext>(                                            \
      const int batch_size,                                                 \
      const int M,                                                          \
      const int N,                                                          \
      const int k,                                                          \
      const int upper,                                                      \
      const T* x,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    const auto nthreads = batch_size * M * N;                               \
    math::Copy(nthreads, x, y, ctx);                                        \
    if (upper > 0) {                                                        \
      _SetTrilu<math::Traits<T>::scalar_type, true>                         \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              nthreads, M, N, k, (math::Traits<T>::scalar_type*)y);         \
    } else {                                                                \
      _SetTrilu<math::Traits<T>::scalar_type, false>                        \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              nthreads, M, N, k, (math::Traits<T>::scalar_type*)y);         \
    }                                                                       \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
