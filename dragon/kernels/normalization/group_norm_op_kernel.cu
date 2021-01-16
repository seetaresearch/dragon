#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

#if __CUDA_ARCH__ >= 350
#define LDG(x, i) __ldg(x + i)
#define LDG2(x, i) convert::To<AccT>(__ldg(x + i))
#else
#define LDG(x, i) x[i]
#define LDG2(x, i) convert::To<AccT>(x[i])
#endif

template <typename T>
__global__ void _GroupNormFusedParams(
    const int N,
    const int G,
    const int D,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* beta,
    T* scale,
    T* bias) {
  const int outer_dim = N * G;
  CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
    const int g = i % G;
    const T mu_val = LDG(mu, i);
    const T rsig_val = LDG(rsig, i);
    CUDA_2D_KERNEL_LOOP2(j, D) {
      const int wi = i * D + j;
      const int gi = g * D + j;
      const T w = LDG(gamma, gi) * rsig_val;
      scale[wi] = w;
      bias[wi] = fma(-w, mu_val, LDG(beta, gi));
    }
  }
}

template <typename T, typename AccT>
__global__ void _GroupNormAffineNCHW(
    const int N,
    const int C,
    const int S,
    const T* x,
    const AccT* scale,
    const AccT* bias,
    T* y) {
  const int outer_dim = N * C;
  CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
    const AccT w = LDG(scale, i);
    const AccT b = LDG(bias, i);
    CUDA_2D_KERNEL_LOOP2(j, S) {
      const int xi = i * S + j;
      y[xi] = convert::To<AccT>(fma(LDG2(x, xi), w, b));
    }
  }
}

template <typename T, typename AccT>
__global__ void _GroupNormAffineNHWC(
    const int N,
    const int C,
    const int S,
    const T* x,
    const AccT* scale,
    const AccT* bias,
    T* y) {
  const int outer_dim = N * S;
  CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
    const int n = i / S;
    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int xi = i * C + j;
      const int wi = n * C + j;
      y[xi] = convert::To<T>(fma(LDG2(x, xi), LDG(scale, wi), LDG(bias, wi)));
    }
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
  const int outer_dim = G * D;
  const int inner_dim = N * S;
  __shared__ typename BlockReduce<AccT>::TempStorage dg_storage;
  __shared__ typename BlockReduce<AccT>::TempStorage db_storage;
  CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
    AccT dg_val = AccT(0), db_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
      const int n = j / S;
      const int xi = kOrder == StorageOrder::NCHW
          ? (n * outer_dim + i) * S + j % S
          : j * outer_dim + i;
      const int mi = n * G + i / D;
      dg_val += LDG2(dy, xi) * (LDG2(x, xi) - LDG(mu, mi)) * LDG(rsig, mi);
      db_val += LDG2(dy, xi);
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
  const int outer_dim = N * G;
  const int inner_dim = D * S;
  __shared__ typename BlockReduce<AccT>::TempStorage ds_storage;
  __shared__ typename BlockReduce<AccT>::TempStorage db_storage;
  CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
    AccT ds_val = AccT(0), db_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
      const int gi = i % G * D + j / S;
      const int xi = kOrder == StorageOrder::NCHW
          ? i * inner_dim + j
          : (i / G * S + j % S) * G * D + gi;
      ds_val += LDG(gamma, gi) * LDG2(dy, xi) * LDG2(x, xi);
      db_val += LDG(gamma, gi) * LDG2(dy, xi);
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
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int mi = kOrder == StorageOrder::NCHW ? i / (D * S)
                                                : i / (C * S) * G + (i / D % G);
    const int gi = kOrder == StorageOrder::NCHW ? (i / S) % C : i % C;
    const AccT u = fma(LDG(db, mi), LDG(mu, mi), -LDG(ds, mi)) *
        (LDG2(x, i) - LDG(mu, mi)) * math::utils::Cube(LDG(rsig, mi));
    const AccT v = LDG(db, mi) * LDG(rsig, mi);
    dx[i] = convert::To<T>(
        LDG(gamma, gi) * LDG2(dy, i) * LDG(rsig, mi) + (u - v) * denom);
  }
}

#undef LDG
#undef LDG2

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_GROUPNORM_KERNEL(name, T, AccT, nblocks, nthreads, ...) \
  if (data_format == "NCHW") {                                           \
    name<T, AccT, StorageOrder::NCHW>                                    \
        <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);     \
  } else if (data_format == "NHWC") {                                    \
    name<T, AccT, StorageOrder::NHWC>                                    \
        <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);     \
  } else {                                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;                 \
  }

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                                     \
  template <>                                                               \
  void GroupNorm<T, AccT, CUDAContext>(                                     \
      const int N,                                                          \
      const int G,                                                          \
      const int D,                                                          \
      const int S,                                                          \
      const string& data_format,                                            \
      const T* x,                                                           \
      const AccT* mu,                                                       \
      const AccT* rsig,                                                     \
      const AccT* gamma,                                                    \
      const AccT* beta,                                                     \
      AccT* scale,                                                          \
      AccT* bias,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    const int C = G * D;                                                    \
    _GroupNormFusedParams<<<                                                \
        CUDA_2D_BLOCKS(N* G),                                               \
        CUDA_THREADS,                                                       \
        0,                                                                  \
        ctx->cuda_stream()>>>(N, G, D, mu, rsig, gamma, beta, scale, bias); \
    if (data_format == "NCHW") {                                            \
      _GroupNormAffineNCHW<<<                                               \
          CUDA_2D_BLOCKS(N* C),                                             \
          CUDA_THREADS,                                                     \
          0,                                                                \
          ctx->cuda_stream()>>>(                                            \
          N,                                                                \
          C,                                                                \
          S,                                                                \
          reinterpret_cast<const math::ScalarType<T>::type*>(x),            \
          scale,                                                            \
          bias,                                                             \
          reinterpret_cast<math::ScalarType<T>::type*>(y));                 \
    } else if (data_format == "NHWC") {                                     \
      _GroupNormAffineNHWC<<<                                               \
          CUDA_2D_BLOCKS(N* C),                                             \
          CUDA_THREADS,                                                     \
          0,                                                                \
          ctx->cuda_stream()>>>(                                            \
          N,                                                                \
          C,                                                                \
          S,                                                                \
          reinterpret_cast<const math::ScalarType<T>::type*>(x),            \
          scale,                                                            \
          bias,                                                             \
          reinterpret_cast<math::ScalarType<T>::type*>(y));                 \
    }                                                                       \
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
    auto nthreads = N * G * D * S;                              \
    DISPATCH_GROUPNORM_KERNEL(                                  \
        _GroupNormWGrad,                                        \
        math::ScalarType<T>::type,                              \
        AccT,                                                   \
        CUDA_2D_BLOCKS(G* D),                                   \
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
        CUDA_2D_BLOCKS(N* G),                                   \
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
        CUDA_BLOCKS(nthreads),                                  \
        CUDA_THREADS,                                           \
        nthreads,                                               \
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

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
