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

/*! BatchNormBackwardTraining <T = ?, Device = CUDA> */

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _BatchNormInternalGrad(
    const int                   N,
    const int                   C,
    const int                   S,
    const Tx*                   x,
    const Tp*                   mu,
    const Tp*                   rsig,
    const Tp*                   gamma,
    const Tx*                   dy,
    Tp*                         ds,
    Tp*                         db,
    Tp*                         dgamma,
    Tp*                         dbeta) {
    const int outer_dim = N * S;
    __shared__ typename BlockReduce<Tp>::TempStorage ds_storage;
    __shared__ typename BlockReduce<Tp>::TempStorage db_storage;
    __shared__ typename BlockReduce<Tp>::TempStorage dga_storage;
    __shared__ typename BlockReduce<Tp>::TempStorage dbe_storage;
    CUDA_2D_KERNEL_LOOP1(i, C) {
        Tp ds_val = Tp(0), db_val = Tp(0);
        Tp dga_val = Tp(0), dbe_val = Tp(0);
        CUDA_2D_KERNEL_LOOP2(j, outer_dim) {
            const int xi = kOrder == StorageOrder::NCHW ?
                (j / S * C + i) * S + j % S : j * C + i;
#if __CUDA_ARCH__ >= 350
            ds_val += L(gamma, i) * L(dy, xi) * L(x, xi);
            db_val += L(gamma, i) * L(dy, xi);
            dga_val += L(dy, xi)
                        * (L(x, xi) - L(mu, i))
                        * L(rsig, i);
            dbe_val += L(dy, xi);
#else
            ds_val += gamma[i] * dy[xi] * x[xi];
            db_val += gamma[i] * dy[xi];
            dga_val += dy[xi] * (x[xi] - mu[i]) * rsig[i];
            dbe_val += dy[xi];
#endif
        }
        ds_val = BlockReduce<Tp>(ds_storage).Reduce(ds_val, cub::Sum());
        db_val = BlockReduce<Tp>(db_storage).Reduce(db_val, cub::Sum());
        dga_val = BlockReduce<Tp>(dga_storage).Reduce(dga_val, cub::Sum());
        dbe_val = BlockReduce<Tp>(dbe_storage).Reduce(dbe_val, cub::Sum());
        if (threadIdx.x == 0) {
            ds[i] = ds_val; db[i] = db_val;
            dgamma[i] = dga_val; dbeta[i] = dbe_val;
        }
    }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _BatchNormTrainingGrad(
    const int                   nthreads,
    const int                   N,
    const int                   C,
    const int                   S,
    const Tx*                   x,
    const Tp*                   mu,
    const Tp*                   rsig,
    const Tp*                   gamma,
    const Tp*                   ds,
    const Tp*                   db,
    const Tx*                   dy,
    Tx*                         dx) {
    const Tp denom = Tp(1) / Tp(N * S);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int pi = kOrder == StorageOrder::NCHW ?
            (i / S) % C : i % C;
#if __CUDA_ARCH__ >= 350
        const Tp u = (L(db, pi) * L(mu, pi) - L(ds, pi))
                        * (L(x, i) - L(mu, pi))
                        * utils::math::Cube(L(rsig, pi));
        const Tp v = L(db, pi) * L(rsig, pi);
        dx[i] = L(gamma, pi) * L(dy, i) * L(rsig, pi)
                        + (u - v) * denom;
#else
        const Tp u = (db[pi] * mu[pi] - ds[pi])
                        * (x[i] - mu[pi])
                        * utils::math::Cube(rsig[pi]);
        const Tp v = db[pi] * rsig[pi];
        dx[i] = gamma[pi] * dy[i] * rsig[pi]
                    + (u - v) * denom;
#endif
    }
}

/*! BatchNormBackwardInference <T = ?, Device = CUDA> */

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _BatchNormWGrad(
    const int                   N,
    const int                   C,
    const int                   S,
    const Tx*                   x,
    const Tp*                   mu,
    const Tp*                   rsig,
    const Tx*                   dy,
    Tp*                         dgamma,
    Tp*                         dbeta) {
    const int outer_dim = N * S;
    __shared__ typename BlockReduce<Tp>::TempStorage dg_storage;
    __shared__ typename BlockReduce<Tp>::TempStorage db_storage;
    CUDA_2D_KERNEL_LOOP1(i, C) {
        Tp dg_val = Tp(0), db_val = Tp(0);
        CUDA_2D_KERNEL_LOOP2(j, outer_dim) {
            const int xi = kOrder == StorageOrder::NCHW ?
                (j / S * C + i) * S + j % S : j * C + i;
#if __CUDA_ARCH__ >= 350
            dg_val += L(dy, xi)
                        * (L(x, xi) - L(mu, i))
                        * L(rsig, i);
            db_val += L(dy, xi);
#else
            dg_val += dy[xi]
                        * (x[xi] - mu[i])
                        * rsig[i];
            db_val += dy[xi];
#endif
        }
        dg_val = BlockReduce<Tp>(dg_storage).Reduce(dg_val, cub::Sum());
        db_val = BlockReduce<Tp>(db_storage).Reduce(db_val, cub::Sum());
        if (threadIdx.x == 0) {
            dgamma[i] = dg_val; dbeta[i] = db_val;
        }
    }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
__global__ void _BatchNormInferenceGrad(
    const int                   nthreads,
    const int                   C,
    const int                   S,
    const Tp*                   rsig,
    const Tp*                   gamma,
    const Tx*                   dy,
    Tx*                         dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int pi = kOrder == StorageOrder::NCHW ?
            (i / S) % C : i % C;
#if __CUDA_ARCH__ >= 350
        dx[i] = L(gamma, pi) * L(dy, i) * L(rsig, pi);
#else
        dx[i] = gamma[pi] * dy[i] * rsig[pi];
#endif
    }
}

/*! Kernel Launchers */

#define DEFINE_BACKWARD_KERNEL_LAUNCHER(Tx, Tp) \
    template <> void BatchNormBackwardTraining<Tx, Tp, CUDAContext>( \
        const int                   N, \
        const int                   C, \
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
        auto nthreads = N * C * S; \
        if (data_format == "NCHW") { \
            _BatchNormInternalGrad<Tx, Tp, StorageOrder::NCHW> \
                <<< CUDA_2D_BLOCKS(C), CUDA_THREADS, \
                    0, ctx->cuda_stream() >>>( \
                N, C, S, x, mu, rsig, gamma, \
                dy, ds, db, dgamma, dbeta \
            ); \
            _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NCHW> \
                <<< CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                    0, ctx->cuda_stream() >>>( \
                nthreads, N, C, S, x, mu, \
                rsig, gamma, ds, db, dy, dx \
            ); \
        } else if (data_format == "NHWC") { \
            _BatchNormInternalGrad<Tx, Tp, StorageOrder::NHWC> \
                <<< CUDA_2D_BLOCKS(C), CUDA_THREADS, \
                    0, ctx->cuda_stream() >>>( \
                N, C, S, x, mu, rsig, gamma, \
                dy, ds, db, dgamma, dbeta \
            ); \
            _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NHWC> \
                <<< CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                    0, ctx->cuda_stream() >>>( \
                nthreads, N, C, S, x, mu, \
                rsig, gamma, ds, db, dy, dx \
            ); \
        } \
    } \
    template <> void BatchNormBackwardInference<Tx, Tp, CUDAContext>( \
        const int                   N, \
        const int                   C, \
        const int                   S, \
        const string&               data_format, \
        const Tx*                   x, \
        const Tp*                   mu, \
        const Tp*                   rsig, \
        const Tp*                   gamma, \
        const Tx*                   dy, \
        Tx*                         dx, \
        Tp*                         dgamma, \
        Tp*                         dbeta, \
        CUDAContext*                ctx) { \
        auto nthreads = N * C * S; \
        if (data_format == "NCHW") { \
            if (dgamma != nullptr) { \
                _BatchNormWGrad<Tx, Tp, StorageOrder::NCHW> \
                    <<< CUDA_2D_BLOCKS(C), CUDA_THREADS, \
                        0, ctx->cuda_stream() >>> \
                    (N, C, S, x, mu, rsig, dy, dgamma, dbeta); \
            } \
            _BatchNormInferenceGrad<Tx, Tp, StorageOrder::NCHW> \
                <<< CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                    0, ctx->cuda_stream() >>> \
                (nthreads, C, S, rsig, gamma, dy, dx); \
        } else if (data_format == "NHWC") { \
            if (dgamma != nullptr) { \
                _BatchNormWGrad<Tx, Tp, StorageOrder::NHWC> \
                    <<< CUDA_2D_BLOCKS(C), CUDA_THREADS, \
                        0, ctx->cuda_stream() >>> \
                    (N, C, S, x, mu, rsig, dy, dgamma, dbeta); \
            } \
            _BatchNormInferenceGrad<Tx, Tp, StorageOrder::NHWC> \
                <<< CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                    0, ctx->cuda_stream() >>> \
                (nthreads, C, S, rsig, gamma, dy, dx); \
        } \
    }

DEFINE_BACKWARD_KERNEL_LAUNCHER(float, float);
#undef L
#undef DEFINE_BACKWARD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namespace dragon

#endif  // WITH_CUDA