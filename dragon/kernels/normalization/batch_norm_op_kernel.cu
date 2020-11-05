#ifdef USE_CUDA

#include "dragon/core/memory.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

#if __CUDA_ARCH__ >= 350
#define LOAD(x, i) __ldg(x + i)
#else
#define LOAD(x, i) x[i]
#endif

namespace {

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _BatchNormExpectation(
    const int N,
    const int C,
    const int S,
    const Tp denorm,
    const Tx* x,
    Tp* ex,
    Tp* ex2) {
  const int outer_dim = N * S;
  __shared__ typename BlockReduce<Tp>::TempStorage ex_storage;
  __shared__ typename BlockReduce<Tp>::TempStorage ex2_storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    Tp ex_val = Tp(0), ex2_val = Tp(0);
    CUDA_2D_KERNEL_LOOP2(j, outer_dim) {
      const int xi = kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                                  : j * C + i;
      ex_val += LOAD(x, xi);
      ex2_val += utils::math::Square(LOAD(x, xi));
    }
    ex_val = BlockReduce<Tp>(ex_storage).Reduce(ex_val, cub::Sum());
    ex2_val = BlockReduce<Tp>(ex2_storage).Reduce(ex2_val, cub::Sum());
    if (threadIdx.x == 0) {
      ex[i] = ex_val * denorm;
      ex2[i] = ex2_val * denorm;
    }
  }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _BatchNormInternalGrad(
    const int N,
    const int C,
    const int S,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tx* dy,
    Tp* dgamma,
    Tp* dbeta) {
  const int outer_dim = N * S;
  __shared__ typename BlockReduce<Tp>::TempStorage dg_storage;
  __shared__ typename BlockReduce<Tp>::TempStorage db_storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    Tp dg_val = Tp(0), db_val = Tp(0);
    CUDA_2D_KERNEL_LOOP2(j, outer_dim) {
      const int xi = kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                                  : j * C + i;
      dg_val += LOAD(dy, xi) * (LOAD(x, xi) - LOAD(mu, i)) * LOAD(rsig, i);
      db_val += LOAD(dy, xi);
    }
    dg_val = BlockReduce<Tp>(dg_storage).Reduce(dg_val, cub::Sum());
    db_val = BlockReduce<Tp>(db_storage).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      dgamma[i] = dg_val;
      dbeta[i] = db_val;
    }
  }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _BatchNormTrainingGrad(
    const int nthreads,
    const int N,
    const int C,
    const int S,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tp* dgamma,
    const Tp* dbeta,
    const Tx* dy,
    Tx* dx) {
  const Tp denom = Tp(1) / Tp(N * S);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int pi = kOrder == StorageOrder::NCHW ? (i / S) % C : i % C;
    const Tp x_norm = (LOAD(x, i) - LOAD(mu, pi)) * LOAD(rsig, pi);
    dx[i] = LOAD(gamma, pi) * LOAD(rsig, pi) *
        (LOAD(dy, i) - fma(x_norm, LOAD(dgamma, pi), LOAD(dbeta, pi)) * denom);
  }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _BatchNormWGrad(
    const int N,
    const int C,
    const int S,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tx* dy,
    Tp* dgamma,
    Tp* dbeta) {
  const int outer_dim = N * S;
  __shared__ typename BlockReduce<Tp>::TempStorage dg_storage;
  __shared__ typename BlockReduce<Tp>::TempStorage db_storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    Tp dg_val = Tp(0), db_val = Tp(0);
    CUDA_2D_KERNEL_LOOP2(j, outer_dim) {
      const int xi = kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                                  : j * C + i;
      dg_val += LOAD(dy, xi) * (LOAD(x, xi) - LOAD(mu, i)) * LOAD(rsig, i);
      db_val += LOAD(dy, xi);
    }
    dg_val = BlockReduce<Tp>(dg_storage).Reduce(dg_val, cub::Sum());
    db_val = BlockReduce<Tp>(db_storage).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      dgamma[i] = dg_val;
      dbeta[i] = db_val;
    }
  }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _BatchNormInferenceGrad(
    const int nthreads,
    const int C,
    const int S,
    const Tp* rsig,
    const Tp* gamma,
    const Tx* dy,
    Tx* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int pi = kOrder == StorageOrder::NCHW ? (i / S) % C : i % C;
    dx[i] = LOAD(gamma, pi) * LOAD(dy, i) * LOAD(rsig, pi);
  }
}

#undef LOAD

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_GRAD_KERNEL_LAUNCHER(Tx, Tp)                                  \
  template <>                                                                \
  void BatchNormExpectation<Tx, Tp, CUDAContext>(                            \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const Tp denorm,                                                       \
      const string& data_format,                                             \
      const Tx* x,                                                           \
      Tp* ex,                                                                \
      Tp* ex2,                                                               \
      CUDAContext* ctx) {                                                    \
    if (data_format == "NCHW") {                                             \
      _BatchNormExpectation<Tx, Tp, StorageOrder::NCHW>                      \
          <<<CUDA_2D_BLOCKS(C), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
              N, C, S, denorm, x, ex, ex2);                                  \
    } else if (data_format == "NHWC") {                                      \
      _BatchNormExpectation<Tx, Tp, StorageOrder::NHWC>                      \
          <<<CUDA_2D_BLOCKS(C), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
              N, C, S, denorm, x, ex, ex2);                                  \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void BatchNormInternalGrad<Tx, Tp, CUDAContext>(                           \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const string& data_format,                                             \
      const Tx* x,                                                           \
      const Tp* mu,                                                          \
      const Tp* rsig,                                                        \
      const Tp* gamma,                                                       \
      const Tx* dy,                                                          \
      Tp* dgamma,                                                            \
      Tp* dbeta,                                                             \
      CUDAContext* ctx) {                                                    \
    if (data_format == "NCHW") {                                             \
      _BatchNormInternalGrad<Tx, Tp, StorageOrder::NCHW>                     \
          <<<CUDA_2D_BLOCKS(C), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
              N, C, S, x, mu, rsig, gamma, dy, dgamma, dbeta);               \
    } else if (data_format == "NHWC") {                                      \
      _BatchNormInternalGrad<Tx, Tp, StorageOrder::NHWC>                     \
          <<<CUDA_2D_BLOCKS(C), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
              N, C, S, x, mu, rsig, gamma, dy, dgamma, dbeta);               \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void BatchNormTrainingGrad<Tx, Tp, CUDAContext>(                           \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const string& data_format,                                             \
      const Tx* x,                                                           \
      const Tp* mu,                                                          \
      const Tp* rsig,                                                        \
      const Tp* gamma,                                                       \
      const Tp* dgamma,                                                      \
      const Tp* dbeta,                                                       \
      const Tx* dy,                                                          \
      Tx* dx,                                                                \
      CUDAContext* ctx) {                                                    \
    const int nthreads = N * C * S;                                          \
    if (data_format == "NCHW") {                                             \
      _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NCHW>                     \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
              nthreads, N, C, S, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    } else if (data_format == "NHWC") {                                      \
      _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NHWC>                     \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
              nthreads, N, C, S, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void BatchNormBackwardTraining<Tx, Tp, CUDAContext>(                       \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const string& data_format,                                             \
      const Tx* x,                                                           \
      const Tp* mu,                                                          \
      const Tp* rsig,                                                        \
      const Tp* gamma,                                                       \
      const Tx* dy,                                                          \
      Tx* dx,                                                                \
      Tp* dgamma,                                                            \
      Tp* dbeta,                                                             \
      CUDAContext* ctx) {                                                    \
    const int nthreads = N * C * S;                                          \
    if (data_format == "NCHW") {                                             \
      _BatchNormInternalGrad<Tx, Tp, StorageOrder::NCHW>                     \
          <<<CUDA_2D_BLOCKS(C), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
              N, C, S, x, mu, rsig, gamma, dy, dgamma, dbeta);               \
      _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NCHW>                     \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
              nthreads, N, C, S, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    } else if (data_format == "NHWC") {                                      \
      _BatchNormInternalGrad<Tx, Tp, StorageOrder::NHWC>                     \
          <<<CUDA_2D_BLOCKS(C), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
              N, C, S, x, mu, rsig, gamma, dy, dgamma, dbeta);               \
      _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NHWC>                     \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
              nthreads, N, C, S, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void BatchNormBackwardInference<Tx, Tp, CUDAContext>(                      \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const string& data_format,                                             \
      const Tx* x,                                                           \
      const Tp* mu,                                                          \
      const Tp* rsig,                                                        \
      const Tp* gamma,                                                       \
      const Tx* dy,                                                          \
      Tx* dx,                                                                \
      Tp* dgamma,                                                            \
      Tp* dbeta,                                                             \
      CUDAContext* ctx) {                                                    \
    const int nthreads = N * C * S;                                          \
    if (data_format == "NCHW") {                                             \
      if (dgamma != nullptr) {                                               \
        _BatchNormWGrad<Tx, Tp, StorageOrder::NCHW>                          \
            <<<CUDA_2D_BLOCKS(C), CUDA_THREADS, 0, ctx->cuda_stream()>>>(    \
                N, C, S, x, mu, rsig, dy, dgamma, dbeta);                    \
      }                                                                      \
      _BatchNormInferenceGrad<Tx, Tp, StorageOrder::NCHW>                    \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
              nthreads, C, S, rsig, gamma, dy, dx);                          \
    } else if (data_format == "NHWC") {                                      \
      if (dgamma != nullptr) {                                               \
        _BatchNormWGrad<Tx, Tp, StorageOrder::NHWC>                          \
            <<<CUDA_2D_BLOCKS(C), CUDA_THREADS, 0, ctx->cuda_stream()>>>(    \
                N, C, S, x, mu, rsig, dy, dgamma, dbeta);                    \
      }                                                                      \
      _BatchNormInferenceGrad<Tx, Tp, StorageOrder::NHWC>                    \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
              nthreads, C, S, rsig, gamma, dy, dx);                          \
    }                                                                        \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float, float);

#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
