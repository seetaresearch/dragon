/*!
 * Codes are based on:
 *
 *    <https://github.com/pytorch/pytorch/blob/master/caffe2/operators/group_norm_op.cu>
 *
 * ------------------------------------------------------------
 */

#ifdef WITH_CUDA

#include "core/mixedmem.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/cub_device.h"

namespace dragon {

namespace kernel {

#define L(x, i) __ldg(x + i)
#define LF(x, i) __half2float(__ldg(x + i))

/*! GroupNormForward <T = ?, Device = CUDA> */

template <typename T>
__global__ void _GroupNormFusedParams(
    const int                   N,
    const int                   G,
    const int                   D,
    const T*                    mu,
    const T*                    rsig,
    const T*                    gamma,
    const T*                    beta,
    T*                          scale,
    T*                          bias) {
    const int outer_dim = N * G;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        const int g = i % G;
#if __CUDA_ARCH__ >= 350
        const T mu_val = L(mu, i);
        const T rsig_val = L(rsig, i);
#else
        const T mu_val = mu[i];
        const T rsig_val = rsig[i];
#endif
        CUDA_2D_KERNEL_LOOP2(j, D) {
            const int i_scale = i * D + j;
            const int i_gamma = g * D + j;
#if __CUDA_ARCH__ >= 350
            const T scale_val = L(gamma, i_gamma) * rsig_val;
            scale[i_scale] = scale_val;
            bias[i_scale] = L(beta, i_gamma) - scale_val * mu_val;
#else
            const T scale_val = gamma[i_gamma] * rsig_val;
            scale[i_scale] = scale_val;
            bias[i_scale] = beta[i_gamma] - scale_val * mu_val;
#endif
        }
    }
}

template <typename Tx, typename Tp>
__global__ void _GroupNormForwardNCHW(
    const int                   N,
    const int                   C,
    const int                   S,
    const Tx*                   x,
    const Tp*                   scale,
    const Tp*                   bias,
    Tx*                         y) {
    const int outer_dim = N * C;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
#if __CUDA_ARCH__ >= 350
        const Tp scale_val = L(scale, i);
        const Tp bias_val = L(bias, i);
#else
        const Tp scale_val = scale[i];
        const Tp bias_val = bias[i];
#endif
        CUDA_2D_KERNEL_LOOP2(j, S) {
            const int idx = i * S + j;
#if __CUDA_ARCH__ >= 350
            y[idx] = L(x, idx) * scale_val + bias_val;
#else
            y[idx] = x[idx] * scale_val + bias_val;
#endif
        }
    }
}

template<> __global__ void _GroupNormForwardNCHW<half, float>(
    const int                   N,
    const int                   C,
    const int                   S,
    const half*                 x,
    const float*                scale,
    const float*                bias,
    half*                       y) {
#if __CUDA_ARCH__ >= 530
    const int outer_dim = N * C;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        const float scale_val = L(scale, i);
        const float bias_val = L(bias, i);
        CUDA_2D_KERNEL_LOOP2(j, S) {
            const int idx = i * S + j;
            y[idx] = __float2half(
                LF(x, idx) * scale_val + bias_val
            );
        }
    }
#endif
}

template <typename Tx, typename Tp>
__global__ void _GroupNormForwardNHWC(
    const int                   N,
    const int                   C,
    const int                   S,
    const Tx*                   x,
    const Tp*                   scale,
    const Tp*                   bias,
    Tx*                         y) {
    const int outer_dim = N * S;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        const int n = i / S;
        CUDA_2D_KERNEL_LOOP2(j, C) {
            const int idx = i * C + j;
            const int i_scale = n * C + j;
#if __CUDA_ARCH__ >= 350
            y[idx] = L(x, idx) * L(scale, i_scale)
                         + L(bias, i_scale);
#else
            y[idx] = x[idx] * scale[i_scale] + bias[i_scale];
#endif
        }
    }
}

template<> __global__ void _GroupNormForwardNHWC<half, float>(
    const int                   N,
    const int                   C,
    const int                   S,
    const half*                 x,
    const float*                scale,
    const float*                bias,
    half*                       y) {
#if __CUDA_ARCH__ >= 530
    const int outer_dim = N * S;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        const int n = i / S;
        CUDA_2D_KERNEL_LOOP2(j, C) {
            const int idx = i * C + j;
            const int i_scale = n * C + j;
            y[idx] = __float2half(
                LF(x, idx) * L(scale, i_scale)
                    + L(bias, i_scale)
            );
        }
    }
#endif
}

/*! GroupNormBackward <T = ?, Device = CUDA> */

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _GroupNormWGrad(
    const int                   N,
    const int                   G,
    const int                   D,
    const int                   S,
    const Tx*                   x,
    const Tp*                   mu,
    const Tp*                   rsig,
    const Tx*                   dy,
    Tp*                         dgamma,
    Tp*                         dbeta) {
    const int outer_dim = G * D;
    const int inner_dim = N * S;
    __shared__ typename BlockReduce<Tp>::TempStorage dg_storage;
    __shared__ typename BlockReduce<Tp>::TempStorage db_storage;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        Tp dg_val = 0, db_val = 0;
        CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
            const int n = j / S;
            const int idx = kOrder == StorageOrder::NCHW ?
                (n * outer_dim + i) * S + j % S :
                 j * outer_dim + i;
            const int i_mu = n * G + i / D;
#if __CUDA_ARCH__ >= 350
            dg_val += L(dy, idx) * (
                L(x, idx) - L(mu, i_mu)
            ) * L(rsig, i_mu);
            db_val += L(dy, idx);
#else
            dg_val += dy[idx] * (x[idx] - mu[i_mu]) * rsig[i_mu];
            db_val += dy[idx];
#endif
        }
        dg_val = BlockReduce<Tp>(dg_storage).Reduce(dg_val, cub::Sum());
        db_val = BlockReduce<Tp>(db_storage).Reduce(db_val, cub::Sum());
        if (threadIdx.x == 0) {
            dgamma[i] = dg_val; dbeta[i] = db_val;
        }
    }
}

template <StorageOrder kOrder>
__global__ void _GroupNormWGradHalf(
    const int                   N,
    const int                   G,
    const int                   D,
    const int                   S,
    const half*                 x,
    const float*                mu,
    const float*                rsig,
    const half*                 dy,
    float*                      dgamma,
    float*                      dbeta) {
#if __CUDA_ARCH__ >= 530
    const int outer_dim = G * D;
    const int inner_dim = N * S;
    __shared__ typename BlockReduce<float>::TempStorage dg_storage;
    __shared__ typename BlockReduce<float>::TempStorage db_storage;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        float dg_val = 0.f, db_val = 0.f;
        CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
            const int n = j / S;
            const int idx = kOrder == StorageOrder::NCHW ?
                (n * outer_dim + i) * S + j % S :
                 j * outer_dim + i;
            const int i_mu = n * G + i / D;
            dg_val += LF(dy, idx) * (
                LF(x, idx) - L(mu, i_mu)
            ) * L(rsig, i_mu);
            db_val += LF(dy, idx);
        }
        dg_val = BlockReduce<float>(dg_storage).Reduce(dg_val, cub::Sum());
        db_val = BlockReduce<float>(db_storage).Reduce(db_val, cub::Sum());
        if (threadIdx.x == 0) {
            dgamma[i] = dg_val; dbeta[i] = db_val;
        }
    }
#endif
}

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _GroupNormInternalGrad(
    const int                   N,
    const int                   G,
    const int                   D,
    const int                   S,
    const Tx*                   x,
    const Tp*                   gamma,
    const Tx*                   dy,
    Tp*                         ds,
    Tp*                         db) {
    const int outer_dim = N * G;
    const int inner_dim = D * S;
    __shared__ typename BlockReduce<Tp>::TempStorage ds_storage;
    __shared__ typename BlockReduce<Tp>::TempStorage db_storage;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        Tp ds_val = 0, db_val = 0;
        CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
            const int i_gamma = i % G * D + j / S;
            const int idx = kOrder == StorageOrder::NCHW
                ? i * inner_dim + j :
                  (i / G * S + j % S) * G * D + i_gamma;
#if __CUDA_ARCH__ >= 350
            ds_val += L(gamma, i_gamma) * L(dy, idx) * L(x, idx);
            db_val += L(gamma, i_gamma) * L(dy, idx);
#else
            ds_val += gamma[i_gamma] * dy[idx] * x[idx];
            db_val += gamma[i_gamma] * dy[idx];
#endif
        }
        ds_val = BlockReduce<Tp>(ds_storage).Reduce(ds_val, cub::Sum());
        db_val = BlockReduce<Tp>(db_storage).Reduce(db_val, cub::Sum());
        if (threadIdx.x == 0) {
            ds[i] = ds_val; db[i] = db_val;
        }
    }
}

template <StorageOrder kOrder>
__global__ void _GroupNormInternalGradHalf(
    const int                   N,
    const int                   G,
    const int                   D,
    const int                   S,
    const half*                 x,
    const float*                gamma,
    const half*                 dy,
    float*                      ds,
    float*                      db) {
#if __CUDA_ARCH__ >= 530
    const int outer_dim = N * G;
    const int inner_dim = D * S;
    __shared__ typename BlockReduce<float>::TempStorage ds_storage;
    __shared__ typename BlockReduce<float>::TempStorage db_storage;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        float ds_val = 0.f, db_val = 0.f;
        CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
            const int i_gamma = i % G * D + j / S;
            const int idx = kOrder == StorageOrder::NCHW
                ? i * inner_dim + j :
                  (i / G * S + j % S) * G * D + i_gamma;
            ds_val += L(gamma, i_gamma) * LF(dy, idx) * LF(x, idx);
            db_val += L(gamma, i_gamma) * LF(dy, idx);
        }
        ds_val = BlockReduce<float>(ds_storage).Reduce(ds_val, cub::Sum());
        db_val = BlockReduce<float>(db_storage).Reduce(db_val, cub::Sum());
        if (threadIdx.x == 0) {
            ds[i] = ds_val; db[i] = db_val;
        }
    }
#endif
}

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _GroupNormGrad(
    const int                   nthreads,
    const int                   G,
    const int                   D,
    const int                   S,
    const Tx*                   x,
    const Tp*                   mu,
    const Tp*                   rsig,
    const Tp*                   gamma,
    const Tp*                   ds,
    const Tp*                   db,
    const Tx*                   dy,
    Tx*                         dx) {
    const int C = G * D;
    const Tp denom = Tp(1) / static_cast<Tp>(D * S);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int i_mu = kOrder == StorageOrder::NCHW ?
            i / (D * S) : i / (C * S) * G + (i / D % G);
        const int i_gamma = kOrder == StorageOrder::NCHW ?
            (i / S) % C : i % C;
#if __CUDA_ARCH__ >= 350
        const Tp u = (
            L(db, i_mu) * L(mu, i_mu) - L(ds, i_mu))
                * (L(x, i) - L(mu, i_mu))
                    * utils::math::Cube<Tp>(L(rsig, i_mu));
        const Tp v = L(db, i_mu) * L(rsig, i_mu);
        dx[i] = L(gamma, i_gamma) * L(dy, i)
            * L(rsig, i_mu) + (u - v) * denom;
#else
        const Tp u = (
            db[i_mu] * mu[i_mu] - ds[i_mu])
                * (x[i] - mu[i_mu])
                    * utils::math::Cube<Tp>(rsig[i_mu]);
        const Tp v = db[i_mu] * rsig[i_mu];
        dx[i] = gamma[i_gamma] * dy[i]
            * rsig[i_mu] + (u - v) * denom;
#endif
    }
}

template <StorageOrder kOrder>
__global__ void _GroupNormGradHalf(
    const int                   nthreads,
    const int                   G,
    const int                   D,
    const int                   S,
    const half*                 x,
    const float*                mu,
    const float*                rsig,
    const float*                gamma,
    const float*                ds,
    const float*                db,
    const half*                 dy,
    half*                       dx) {
#if __CUDA_ARCH__ >= 530
    const int C = G * D;
    const float denom = 1.f / static_cast<float>(D * S);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int i_mu = kOrder == StorageOrder::NCHW ?
            i / (D * S) : i / (C * S) * G + (i / D % G);
        const int i_gamma = kOrder == StorageOrder::NCHW ?
            (i / S) % C : i % C;
        const float u = (
            L(db, i_mu) * L(mu, i_mu) - L(ds, i_mu))
                * (LF(x, i) - L(mu, i_mu))
                    * utils::math::Cube<float>(L(rsig, i_mu));
        const float v = L(db, i_mu) * L(rsig, i_mu);
        dx[i] = __float2half(
            L(gamma, i_gamma) * LF(dy, i)
                * L(rsig, i_mu) + (u - v) * denom
        );
    }
#endif
}

/*! Kernel Launchers */

#define DEFINE_FORWARD_KERNEL_LAUNCHER(Tx, Tp) \
    template <> void GroupNormForward<Tx, Tp, CUDAContext>( \
        const int                   N, \
        const int                   G, \
        const int                   D, \
        const int                   S, \
        const string&               data_format, \
        const Tx*                   x, \
        const Tp*                   mu, \
        const Tp*                   rsig, \
        const Tp*                   gamma, \
        const Tp*                   beta, \
        Tp*                         scale, \
        Tp*                         bias, \
        Tx*                         y, \
        CUDAContext*                ctx) { \
        const int C = G * D; \
        _GroupNormFusedParams<Tp> \
            << < CUDA_2D_BLOCKS(N * G), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (N, G, D, mu, rsig, gamma, beta, scale, bias); \
        if (data_format == "NCHW") { \
            _GroupNormForwardNCHW<Tx, Tp> \
                << < CUDA_2D_BLOCKS(N * C), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (N, C, S, x, scale, bias, y); \
        } else if (data_format == "NHWC") { \
            _GroupNormForwardNHWC<Tx, Tp> \
                << < CUDA_2D_BLOCKS(N * C), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (N, C, S, x, scale, bias, y); \
        } \
    }

#define DEFINE_BACKWARD_KERNEL_LAUNCHER(Tx, Tp) \
    template <> void GroupNormBackward<Tx, Tp, CUDAContext>( \
        const int                   N, \
        const int                   G, \
        const int                   D, \
        const int                   S, \
        const string&               data_format, \
        const Tx*                   x, \
        const Tp*                   mu, \
        const Tp*                   rsig, \
        const Tp*                   gamma, \
        const Tx*                   dy, \
        Tp*                         ds, \
        Tp*                         db, \
        Tx*                         dx, \
        Tp*                         dgamma, \
        Tp*                         dbeta, \
        CUDAContext*                ctx) { \
        auto nthreads = N * G * D * S; \
        if (data_format == "NCHW") { \
            _GroupNormWGrad<Tx, Tp, StorageOrder::NCHW> \
                << < CUDA_2D_BLOCKS(G * D), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (N, G, D, S, x, mu, rsig, dy, dgamma, dbeta); \
            _GroupNormInternalGrad<Tx, Tp, StorageOrder::NCHW> \
                << < CUDA_2D_BLOCKS(N * G), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (N, G, D, S, x, gamma, dy, ds, db); \
            _GroupNormGrad<Tx, Tp, StorageOrder::NCHW> \
                << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (nthreads, G, D, S, x, mu, rsig, gamma, ds, db, dy, dx); \
        } else if (data_format == "NHWC") { \
            _GroupNormWGrad<Tx, Tp, StorageOrder::NHWC> \
                << < CUDA_2D_BLOCKS(G * D), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (N, G, D, S, x, mu, rsig, dy, dgamma, dbeta); \
            _GroupNormInternalGrad<Tx, Tp, StorageOrder::NHWC> \
                << < CUDA_2D_BLOCKS(N * G), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (N, G, D, S, x, gamma, dy, ds, db); \
            _GroupNormGrad<Tx, Tp, StorageOrder::NHWC> \
                << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (nthreads, G, D, S, x, mu, rsig, gamma, ds, db, dy, dx); \
        } \
    }

DEFINE_FORWARD_KERNEL_LAUNCHER(float, float);
DEFINE_BACKWARD_KERNEL_LAUNCHER(float, float);

template <> void GroupNormForward<float16, float, CUDAContext>(
    const int                   N,
    const int                   G,
    const int                   D,
    const int                   S,
    const string&               data_format,
    const float16*              x,
    const float*                mu,
    const float*                rsig,
    const float*                gamma,
    const float*                beta,
    float*                      scale,
    float*                      bias,
    float16*                    y,
    CUDAContext*                ctx) {
    const int C = G * D;
    _GroupNormFusedParams<float>
        << < CUDA_2D_BLOCKS(N * G), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (N, G, D, mu, rsig, gamma, beta, scale, bias);
    if (data_format == "NCHW") {
        _GroupNormForwardNCHW<half, float>
            << < CUDA_2D_BLOCKS(N * C), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (N, C, S, reinterpret_cast<const half*>(x),
                scale, bias, reinterpret_cast<half*>(y));
    } else if (data_format == "NHWC") {
        _GroupNormForwardNHWC<half, float>
            << < CUDA_2D_BLOCKS(N * C), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (N, C, S, reinterpret_cast<const half*>(x),
                scale, bias, reinterpret_cast<half*>(y));
    }
}

template <> void GroupNormBackward<float16, float, CUDAContext>(
    const int                   N,
    const int                   G,
    const int                   D,
    const int                   S,
    const string&               data_format,
    const float16*              x,
    const float*                mu,
    const float*                rsig,
    const float*                gamma,
    const float16*              dy,
    float*                      ds,
    float*                      db,
    float16*                    dx,
    float*                      dgamma,
    float*                      dbeta,
    CUDAContext*                ctx) {
    auto nthreads = N * G * D * S;
    if (data_format == "NCHW") {
        _GroupNormWGradHalf<StorageOrder::NCHW>
            << < CUDA_2D_BLOCKS(G * D), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (N, G, D, S, reinterpret_cast<const half*>(x),
                mu, rsig, reinterpret_cast<const half*>(dy),
                    dgamma, dbeta);
        _GroupNormInternalGradHalf<StorageOrder::NCHW>
            << < CUDA_2D_BLOCKS(N * G), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (N, G, D, S, reinterpret_cast<const half*>(x),
                gamma, reinterpret_cast<const half*>(dy),
                    ds, db);
        _GroupNormGradHalf<StorageOrder::NCHW>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> > \
            (nthreads, G, D, S, reinterpret_cast<const half*>(x),
                mu, rsig, gamma, ds, db,
                    reinterpret_cast<const half*>(dy),
                        reinterpret_cast<half*>(dx));
    } else if (data_format == "NHWC") { \
        _GroupNormWGradHalf<StorageOrder::NHWC>
            << < CUDA_2D_BLOCKS(G * D), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (N, G, D, S, reinterpret_cast<const half*>(x),
                mu, rsig, reinterpret_cast<const half*>(dy),
                    dgamma, dbeta);
        _GroupNormInternalGradHalf<StorageOrder::NHWC>
            << < CUDA_2D_BLOCKS(N * G), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (N, G, D, S, reinterpret_cast<const half*>(x),
                gamma, reinterpret_cast<const half*>(dy),
                    ds, db);
        _GroupNormGradHalf<StorageOrder::NHWC>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> > \
            (nthreads, G, D, S, reinterpret_cast<const half*>(x),
                mu, rsig, gamma, ds, db,
                    reinterpret_cast<const half*>(dy),
                        reinterpret_cast<half*>(dx));
    }
}

#undef L
#undef LF
#undef DEFINE_FORWARD_KERNEL_LAUNCHER
#undef DEFINE_BACKWARD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namespace dragon

#endif  // WITH_CUDA