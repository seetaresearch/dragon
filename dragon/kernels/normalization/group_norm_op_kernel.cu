#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

#define LDG(x, i) __ldg(x + i)
#define LDG2(x, i) convert::To<AccT>(__ldg(x + i))

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
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int ng = kOrder == StorageOrder::NCHW ? i / (D * S)
                                                : i / (C * S) * G + (i / D % G);
    const int c = kOrder == StorageOrder::NCHW ? i / S % C : i % C;
    y[i] = convert::To<T>(
        fma((convert::To<AccT>(x[i]) - __ldg(mu + ng)) * __ldg(rsig + ng),
            __ldg(gamma + c),
            __ldg(beta + c)));
  }
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
    CUDA_2D_KERNEL_LOOP2(j, NxS) {
      const int n = j / S;
      const int ng = n * G + i / D;
      const int idx = kOrder == StorageOrder::NCHW ? (n * GxD + i) * S + j % S
                                                   : j * GxD + i;
      dg_val += LDG2(dy, idx) * (LDG2(x, idx) - LDG(mu, ng)) * LDG(rsig, ng);
      db_val += LDG2(dy, idx);
    }
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
    CUDA_2D_KERNEL_LOOP2(j, DxS) {
      const int c = i % G * D + j / S;
      const int idx = kOrder == StorageOrder::NCHW
          ? i * DxS + j
          : (i / G * S + j % S) * G * D + c;
      ds_val += LDG(gamma, c) * LDG2(dy, idx) * LDG2(x, idx);
      db_val += LDG(gamma, c) * LDG2(dy, idx);
    }
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
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int ng = kOrder == StorageOrder::NCHW ? i / (D * S)
                                                : i / (C * S) * G + (i / D % G);
    const int c = kOrder == StorageOrder::NCHW ? i / S % C : i % C;
    const AccT u = fma(LDG(db, ng), LDG(mu, ng), -LDG(ds, ng)) *
        (LDG2(x, i) - LDG(mu, ng)) * math::utils::Cube(LDG(rsig, ng));
    const AccT v = LDG(db, ng) * LDG(rsig, ng);
    dx[i] = convert::To<T>(
        LDG(gamma, c) * LDG2(dy, i) * LDG(rsig, ng) + (u - v) * denom);
  }
}

#undef LDG
#undef LDG2

} // namespace

/* ------------------- Launcher Separator ------------------- */

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

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                        \
  template <>                                                  \
  void GroupNorm<T, AccT, CUDAContext>(                        \
      const int N,                                             \
      const int G,                                             \
      const int D,                                             \
      const int S,                                             \
      const string& data_format,                               \
      const T* x,                                              \
      const AccT* mu,                                          \
      const AccT* rsig,                                        \
      const AccT* gamma,                                       \
      const AccT* beta,                                        \
      T* y,                                                    \
      CUDAContext* ctx) {                                      \
    const auto NxCxS = N * G * D * S;                          \
    DISPATCH_GROUPNORM_KERNEL(                                 \
        _GroupNorm,                                            \
        math::ScalarType<T>::type,                             \
        AccT,                                                  \
        CUDA_BLOCKS(NxCxS),                                    \
        CUDA_THREADS,                                          \
        NxCxS,                                                 \
        G,                                                     \
        D,                                                     \
        S,                                                     \
        reinterpret_cast<const math::ScalarType<T>::type*>(x), \
        mu,                                                    \
        rsig,                                                  \
        gamma,                                                 \
        beta,                                                  \
        reinterpret_cast<math::ScalarType<T>::type*>(y));      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                    \
  template <>                                                   \
  void GroupNormGrad<T, AccT, CUDAContext>(                     \
      const int N,                                              \
      const int G,                                              \
      const int D,                                              \
      const int S,                                              \
      const string& data_format,                                \
      const T* x,                                               \
      const AccT* mu,                                           \
      const AccT* rsig,                                         \
      const AccT* gamma,                                        \
      const T* dy,                                              \
      AccT* ds,                                                 \
      AccT* db,                                                 \
      AccT* dgamma,                                             \
      AccT* dbeta,                                              \
      T* dx,                                                    \
      CUDAContext* ctx) {                                       \
    const auto NxCxS = N * G * D * S;                           \
    DISPATCH_GROUPNORM_KERNEL(                                  \
        _GroupNormWGrad,                                        \
        math::ScalarType<T>::type,                              \
        AccT,                                                   \
        G* D,                                                   \
        CUDA_THREADS,                                           \
        N,                                                      \
        G,                                                      \
        D,                                                      \
        S,                                                      \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),  \
        mu,                                                     \
        rsig,                                                   \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy), \
        dgamma,                                                 \
        dbeta);                                                 \
    DISPATCH_GROUPNORM_KERNEL(                                  \
        _GroupNormInternalGrad,                                 \
        math::ScalarType<T>::type,                              \
        AccT,                                                   \
        N* G,                                                   \
        CUDA_THREADS,                                           \
        N,                                                      \
        G,                                                      \
        D,                                                      \
        S,                                                      \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),  \
        gamma,                                                  \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy), \
        ds,                                                     \
        db);                                                    \
    DISPATCH_GROUPNORM_KERNEL(                                  \
        _GroupNormGrad,                                         \
        math::ScalarType<T>::type,                              \
        AccT,                                                   \
        CUDA_BLOCKS(NxCxS),                                     \
        CUDA_THREADS,                                           \
        NxCxS,                                                  \
        G,                                                      \
        D,                                                      \
        S,                                                      \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),  \
        mu,                                                     \
        rsig,                                                   \
        gamma,                                                  \
        ds,                                                     \
        db,                                                     \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy), \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));      \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_GROUPNORM_KERNEL

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
