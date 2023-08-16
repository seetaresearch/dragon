#include "dragon/kernels/normalization/op_kernels.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _GroupNorm(
    const int NxCxS,
    const int G,
    const int D,
    const int S,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* beta,
    T* y) {
  const int C = G * D;
  CUDA_1D_KERNEL_LOOP(i, NxCxS) { // clang-format off
    const int ng = kOrder == StorageOrder::NCHW ? i / (D * S)
                                                : i / (C * S) * G + (i / D % G);
    const int c = kOrder == StorageOrder::NCHW ? i / S % C : i % C;
    y[i] = fma((convert::To<AccT>(x[i]) - __ldg(mu + ng)) * __ldg(rsig + ng),
                __ldg(gamma + c),  __ldg(beta + c));
  } // clang-format on
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _GroupNormWGrad(
    const int N,
    const int G,
    const int D,
    const int S,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const T* dy,
    AccT* dgamma,
    AccT* dbeta) {
  const int GxD = G * D;
  const int NxS = N * S;
  __shared__ typename BlockReduce<AccT>::TempStorage dg_storage;
  __shared__ typename BlockReduce<AccT>::TempStorage db_storage;
  CUDA_2D_KERNEL_LOOP1(i, GxD) {
    AccT dg_val = AccT(0), db_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, NxS) { // clang-format off
      const int n = j / S;
      const int ng = n * G + i / D;
      const int idx = kOrder == StorageOrder::NCHW ? (n * GxD + i) * S + j % S
                                                   : j * GxD + i;
      dg_val += math::utils::LDGC<AccT>(dy + idx) * (
          (convert::To<AccT>(x[idx]) - __ldg(mu + ng)) *  __ldg(rsig + ng));
      db_val += math::utils::LDGC<AccT>(dy + idx);
    } // clang-format on
    dg_val = BlockReduce<AccT>(dg_storage).Sum(dg_val);
    db_val = BlockReduce<AccT>(db_storage).Sum(db_val);
    if (threadIdx.x == 0) {
      dgamma[i] = dg_val;
      dbeta[i] = db_val;
    }
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _GroupNormInternalGrad(
    const int N,
    const int G,
    const int D,
    const int S,
    const T* x,
    const AccT* gamma,
    const T* dy,
    AccT* ds,
    AccT* db) {
  const int NxG = N * G;
  const int DxS = D * S;
  __shared__ typename BlockReduce<AccT>::TempStorage ds_storage;
  __shared__ typename BlockReduce<AccT>::TempStorage db_storage;
  CUDA_2D_KERNEL_LOOP1(i, NxG) {
    AccT ds_val = AccT(0), db_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, DxS) { // clang-format off
      const int c = i % G * D + j / S;
      const int idx = kOrder == StorageOrder::NCHW
          ? i * DxS + j
          : (i / G * S + j % S) * G * D + c;
      db_val += __ldg(gamma + c) * math::utils::LDGC<AccT>(dy + idx);
      ds_val += __ldg(gamma + c) * math::utils::LDGC<AccT>(dy + idx)
                                 * convert::To<AccT>(x[idx]);
    } // clang-format on
    ds_val = BlockReduce<AccT>(ds_storage).Sum(ds_val);
    db_val = BlockReduce<AccT>(db_storage).Sum(db_val);
    if (threadIdx.x == 0) {
      ds[i] = ds_val;
      db[i] = db_val;
    }
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _GroupNormGrad(
    const int NxCxS,
    const int G,
    const int D,
    const int S,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* ds,
    const AccT* db,
    const T* dy,
    T* dx) {
  const int C = G * D;
  const AccT denom = AccT(1) / AccT(D * S);
  CUDA_1D_KERNEL_LOOP(i, NxCxS) { // clang-format off
    const int ng = kOrder == StorageOrder::NCHW ? i / (D * S)
                                                : i / (C * S) * G + (i / D % G);
    const int c = kOrder == StorageOrder::NCHW ? i / S % C : i % C;
    const AccT u = fma(__ldg(db + ng), __ldg(mu + ng), -__ldg(ds + ng))
                      * (convert::To<AccT>(x[i]) - __ldg(mu + ng))
                      * math::utils::Cube(__ldg(rsig + ng));
    const AccT v = __ldg(db + ng) * __ldg(rsig + ng);
    dx[i] = convert::To<AccT>(dy[i]) * __ldg(gamma + c) * __ldg(rsig + ng) + (u - v) * denom;
  } // clang-format on
}

} // namespace

#define DISPATCH_GROUPNORM_KERNEL(name, T, AccT, kBlocks, kThreads, ...) \
  if (data_format == "NCHW") {                                           \
    name<T, AccT, StorageOrder::NCHW>                                    \
        <<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);     \
  } else if (data_format == "NHWC") {                                    \
    name<T, AccT, StorageOrder::NHWC>                                    \
        <<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);     \
  } else {                                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;                 \
  }

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                           \
  template <>                                                     \
  void GroupNorm<T, AccT, CUDAContext>(                           \
      const int N,                                                \
      const int G,                                                \
      const int D,                                                \
      const int S,                                                \
      const string& data_format,                                  \
      const T* x,                                                 \
      const AccT* mu,                                             \
      const AccT* rsig,                                           \
      const AccT* gamma,                                          \
      const AccT* beta,                                           \
      T* y,                                                       \
      CUDAContext* ctx) {                                         \
    const auto NxCxS = N * G * D * S;                             \
    DISPATCH_GROUPNORM_KERNEL(                                    \
        _GroupNorm,                                               \
        math::Traits<T>::scalar_type,                             \
        AccT,                                                     \
        CUDA_BLOCKS(NxCxS),                                       \
        CUDA_THREADS,                                             \
        NxCxS,                                                    \
        G,                                                        \
        D,                                                        \
        S,                                                        \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x), \
        mu,                                                       \
        rsig,                                                     \
        gamma,                                                    \
        beta,                                                     \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                       \
  template <>                                                      \
  void GroupNormGrad<T, AccT, CUDAContext>(                        \
      const int N,                                                 \
      const int G,                                                 \
      const int D,                                                 \
      const int S,                                                 \
      const string& data_format,                                   \
      const T* x,                                                  \
      const AccT* mu,                                              \
      const AccT* rsig,                                            \
      const AccT* gamma,                                           \
      const T* dy,                                                 \
      AccT* ds,                                                    \
      AccT* db,                                                    \
      AccT* dgamma,                                                \
      AccT* dbeta,                                                 \
      T* dx,                                                       \
      CUDAContext* ctx) {                                          \
    db = ds == db ? ds + N * G : db;                               \
    const auto NxCxS = N * G * D * S;                              \
    DISPATCH_GROUPNORM_KERNEL(                                     \
        _GroupNormWGrad,                                           \
        math::Traits<T>::scalar_type,                              \
        AccT,                                                      \
        G* D,                                                      \
        CUDA_THREADS,                                              \
        N,                                                         \
        G,                                                         \
        D,                                                         \
        S,                                                         \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),  \
        mu,                                                        \
        rsig,                                                      \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy), \
        dgamma,                                                    \
        dbeta);                                                    \
    DISPATCH_GROUPNORM_KERNEL(                                     \
        _GroupNormInternalGrad,                                    \
        math::Traits<T>::scalar_type,                              \
        AccT,                                                      \
        N* G,                                                      \
        CUDA_THREADS,                                              \
        N,                                                         \
        G,                                                         \
        D,                                                         \
        S,                                                         \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),  \
        gamma,                                                     \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy), \
        ds,                                                        \
        db);                                                       \
    DISPATCH_GROUPNORM_KERNEL(                                     \
        _GroupNormGrad,                                            \
        math::Traits<T>::scalar_type,                              \
        AccT,                                                      \
        CUDA_BLOCKS(NxCxS),                                        \
        CUDA_THREADS,                                              \
        NxCxS,                                                     \
        G,                                                         \
        D,                                                         \
        S,                                                         \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),  \
        mu,                                                        \
        rsig,                                                      \
        gamma,                                                     \
        ds,                                                        \
        db,                                                        \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy), \
        reinterpret_cast<math::Traits<T>::scalar_type*>(dx));      \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_GROUPNORM_KERNEL

} // namespace kernels

} // namespace dragon
