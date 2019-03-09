/*!
 * Codes are based on:
 *
 *    <https://github.com/pytorch/pytorch/blob/master/caffe2/operators/group_norm_op.cc>
 *
 * ------------------------------------------------------------
 */

#include "core/mixedmem.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/eigen_utils.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

template <typename T>
void _GroupNormFusedParams(
    const int                   N,
    const int                   G,
    const int                   D,
    const T*                    mu,
    const T*                    rsig,
    const T*                    gamma,
    const T*                    beta,
    T*                          scale,
    T*                          bias) {
    const int C = G * D;
    ConstEigenArrayMap<T> gamma_arr(gamma, D, G);
    ConstEigenArrayMap<T> beta_arr(beta, D, G);
    for (int i = 0; i < N; ++i) {
        EigenArrayMap<T> scale_arr(scale + i * C, D, G);
        scale_arr = gamma_arr.rowwise() *
            ConstEigenVectorArrayMap<T>(rsig + i * G, G).transpose();
        EigenArrayMap<T>(bias + i * C, D, G) = beta_arr -
                    scale_arr.rowwise() *
            ConstEigenVectorArrayMap<T>(mu + i * G, G).transpose();
    }
}

template <typename Tx, typename Tp>
void _GroupNormForwardNCHW(
    const int                   N,
    const int                   C,
    const int                   S,
    const Tx*                   x,
    const Tp*                   scale,
    const Tp*                   bias,
    Tx*                         y) {
    EigenArrayMap<Tx>(y, S, N * C) =
        (ConstEigenArrayMap<Tx>(x, S, N * C).rowwise() *
            ConstEigenVectorArrayMap<Tp>(scale, N * C).transpose())
        .rowwise() + ConstEigenVectorArrayMap<Tp>(bias, N * C).transpose();
}

template <typename Tx, typename Tp>
void _GroupNormForwardNHWC(
    const int                   N,
    const int                   C,
    const int                   S,
    const Tx*                   x,
    const Tp*                   scale,
    const Tp*                   bias,
    Tx*                         y) {
    const int SC = S * C;
    for (int i = 0; i < N; ++i) {
        EigenArrayMap<Tx>(y + i * SC, C, S) =
            (ConstEigenArrayMap<Tx>(x + i * SC, C, S).colwise() *
                ConstEigenVectorArrayMap<Tp>(scale + i * C, C))
            .colwise() + ConstEigenVectorArrayMap<Tp>(bias + i * C, C);
    }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
void _GroupNormInternalGrad(
    const std::array<int, 4>&   dims,
    const Tx*                   x,
    const Tp*                   gamma,
    const Tx*                   dy,
    Tp*                         ds,
    Tp*                         db) {
    const int kGDim = kOrder == StorageOrder::NCHW ? 1 : 2;
    const int kDDim = kOrder == StorageOrder::NCHW ? 2 : 3;
    const int count = dims[0] * dims[1] * dims[2] * dims[3];
    std::array<int, 4> idx = { 0, 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        const int i_mu = idx[0] * dims[kGDim] + idx[kGDim];
        const int i_gamma = idx[kGDim] * dims[kDDim] + idx[kDDim];
        ds[i_mu] += gamma[i_gamma] * dy[i] * x[i];
        db[i_mu] += gamma[i_gamma] * dy[i];
        utils::IncreaseIndexInDims(4, dims.data(), idx.data());
    }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
void _GroupNormGrad(
    const std::array<int, 4>&   dims,
    const Tx*                   x,
    const Tp*                   mu,
    const Tp*                   rsig,
    const Tp*                   gamma,
    const Tp*                   ds,
    const Tp*                   db,
    const Tx*                   dy,
    Tx*                         dx,
    Tp*                         dgamma,
    Tp*                         dbeta) {
    const int kGDim = kOrder == StorageOrder::NCHW ? 1 : 2;
    const int kDDim = kOrder == StorageOrder::NCHW ? 2 : 3;
    const int count = dims[0] * dims[1] * dims[2] * dims[3];
    const int S = kOrder == StorageOrder::NCHW ? dims[3] : dims[1];
    const Tp denom = Tp(1) / static_cast<Tp>(dims[kDDim] * S);
    std::array<int, 4> idx = { 0, 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        const int i_mu = idx[0] * dims[kGDim] + idx[kGDim];
        const int i_gamma = idx[kGDim] * dims[kDDim] + idx[kDDim];
        const Tp u = (db[i_mu] * mu[i_mu] - ds[i_mu]) *
            (x[i] - mu[i_mu]) * utils::math::Cube(rsig[i_mu]);
        const Tp v = db[i_mu] * rsig[i_mu];
        dx[i] = gamma[i_gamma] * dy[i] * rsig[i_mu] + (u - v) * denom;
        dgamma[i_gamma] += dy[i] * (x[i] - mu[i_mu]) * rsig[i_mu];
        dbeta[i_gamma] += dy[i];
        utils::IncreaseIndexInDims(4, dims.data(), idx.data());
    }
}

/*! Kernel Launchers */

#define DEFINE_FORWARD_KERNEL_LAUNCHER(Tx, Tp) \
    template <> void GroupNormForward<Tx, Tp, CPUContext>( \
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
        CPUContext*                 ctx) { \
        const int C = G * D; \
        _GroupNormFusedParams<Tp>(N, G, D, \
            mu, rsig, gamma, beta, scale, bias); \
        if (data_format == "NCHW") { \
            _GroupNormForwardNCHW<Tx, Tp>( \
                N, C, S, x, scale, bias, y); \
        } else if (data_format == "NHWC") { \
            _GroupNormForwardNHWC<Tx, Tp>( \
                N, C, S, x, scale, bias, y); \
        } \
    }

#define DEFINE_BACKWARD_KERNEL_LAUNCHER(Tx, Tp) \
    template <> void GroupNormBackward<Tx, Tp, CPUContext>( \
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
        CPUContext*                 ctx) { \
        math::Set(N * G, (Tp)0, ds, ctx); \
        math::Set(N * G, (Tp)0, db, ctx); \
        math::Set(G * D, (Tp)0, dgamma, ctx); \
        math::Set(G * D, (Tp)0, dbeta, ctx); \
        if (data_format == "NCHW") { \
            _GroupNormInternalGrad<Tx, Tp, StorageOrder::NCHW>( \
                { N, G, D, S }, x, gamma, dy, ds, db); \
            _GroupNormGrad<Tx, Tp, StorageOrder::NCHW>( \
                { N, G, D, S }, x, mu, rsig, gamma, \
                    ds, db, dy, dx, dgamma, dbeta); \
        } else if (data_format == "NHWC") { \
            _GroupNormInternalGrad<Tx, Tp, StorageOrder::NHWC>( \
                { N, S, G, D }, x, gamma, dy, ds, db); \
            _GroupNormGrad<Tx, Tp, StorageOrder::NHWC>( \
                { N, S, G, D }, x, mu, rsig, gamma, \
                    ds, db, dy, dx, dgamma, dbeta); \
        } \
    }

DEFINE_FORWARD_KERNEL_LAUNCHER(float, float);
DEFINE_BACKWARD_KERNEL_LAUNCHER(float, float);

template <> void GroupNormForward<float16, float, CPUContext>(
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
    CPUContext*                 ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void GroupNormBackward<float16, float, CPUContext>(
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
    CPUContext*                 ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_FORWARD_KERNEL_LAUNCHER
#undef DEFINE_BACKWARD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namespace dragon