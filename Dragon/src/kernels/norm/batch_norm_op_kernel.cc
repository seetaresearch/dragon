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

/*! BatchNormBackwardTraining <T = ?, Device = CPU> */

template <typename Tx, typename Tp, StorageOrder kOrder>
void _BatchNormInternalGrad(
    const std::array<int, 3>&   dims,
    const Tx*                   x,
    const Tp*                   mu,
    const Tp*                   rsig,
    const Tp*                   gamma,
    const Tx*                   dy,
    Tp*                         ds,
    Tp*                         db,
    Tp*                         dgamma,
    Tp*                         dbeta) {
    const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
    const int count = dims[0] * dims[1] * dims[2];
    std::array<int, 3> idx = { 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        const int pi = idx[kCDim];
        ds[pi] += gamma[pi] * dy[i] * x[i];
        db[pi] += gamma[pi] * dy[i];
        dgamma[pi] += dy[i] * (x[i] - mu[pi]) * rsig[pi];
        dbeta[pi] += dy[i];
        utils::IncreaseIndexInDims(3, dims.data(), idx.data());
    }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
void _BatchNormTrainingGrad(
    const std::array<int, 3>&   dims,
    const Tx*                   x,
    const Tp*                   mu,
    const Tp*                   rsig,
    const Tp*                   gamma,
    const Tp*                   ds,
    const Tp*                   db,
    const Tx*                   dy,
    Tx*                         dx) {
    const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
    const int count = dims[0] * dims[1] * dims[2];
    const Tp denom = Tp(1) / static_cast<Tp>(count / dims[kCDim]);
    std::array<int, 3> idx = { 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        const int pi = idx[kCDim];
        const Tp u = (db[pi] * mu[pi] - ds[pi]) *
            (x[i] - mu[pi]) * utils::math::Cube(rsig[pi]);
        const Tp v = db[pi] * rsig[pi];
        dx[i] = gamma[pi] * dy[i] * rsig[pi] + (u - v) * denom;
        utils::IncreaseIndexInDims(3, dims.data(), idx.data());
    }
}

/*! BatchNormBackwardInference <T = ?, Device = CPU> */

template <typename Tx, typename Tp, StorageOrder kOrder>
void _BatchNormWGrad(
    const std::array<int, 3>&   dims,
    const Tx*                   x,
    const Tp*                   mu,
    const Tp*                   rsig,
    const Tx*                   dy,
    Tp*                         dgamma,
    Tp*                         dbeta) {
    const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
    const int count = dims[0] * dims[1] * dims[2];
    std::array<int, 3> idx = { 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        const int pi = idx[kCDim];
        dgamma[pi] += dy[i] * (x[i] - mu[pi]) * rsig[pi];
        dbeta[pi] += dy[i];
        utils::IncreaseIndexInDims(3, dims.data(), idx.data());
    }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
void _BatchNormInferenceGrad(
    const int                   N,
    const int                   C,
    const int                   S,
    const Tp*                   rsig,
    const Tp*                   gamma,
    const Tx*                   dy,
    Tx*                         dx) {
    if (kOrder == StorageOrder::NCHW) {
        const int CS = C * S;
        for (int i = 0; i < N; ++i) {
            EigenArrayMap<Tx>(dx + i * CS, S, C) =
                (ConstEigenArrayMap<Tx>(dy + i * CS, S, C).rowwise() *
                    (ConstEigenVectorArrayMap<Tp>(gamma, C) *
                        ConstEigenVectorArrayMap<Tp>(rsig, C)).transpose());
        }
    } else if (kOrder == StorageOrder::NHWC) {
        EigenArrayMap<Tx>(dx, C, N * S) =
            (ConstEigenArrayMap<Tx>(dy, C, N * S).colwise() *
                (ConstEigenVectorArrayMap<Tp>(gamma, C) *
                    ConstEigenVectorArrayMap<Tp>(rsig, C)));
    }
}

/*! Kernel Launchers */

#define DEFINE_BACKWARD_KERNEL_LAUNCHER(Tx, Tp) \
    template <> void BatchNormBackwardTraining<Tx, Tp, CPUContext>( \
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
        CPUContext*                 ctx) { \
        math::Set(C, Tp(0), ds, ctx); \
        math::Set(C, Tp(0), db, ctx); \
        math::Set(C, Tp(0), dgamma, ctx); \
        math::Set(C, Tp(0), dbeta, ctx); \
        if (data_format == "NCHW") { \
            _BatchNormInternalGrad<Tx, Tp, StorageOrder::NCHW>( \
                { N, C, S }, x, mu, rsig, gamma, \
                dy, ds, db, dgamma, dbeta \
            ); \
            _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NCHW>( \
                { N, C, S }, x, mu, rsig, gamma, ds, db, dy, dx \
            ); \
        } else if (data_format == "NHWC") { \
            _BatchNormInternalGrad<Tx, Tp, StorageOrder::NHWC>( \
                { N, S, C }, x, mu, rsig, gamma, \
                dy, ds, db, dgamma, dbeta \
            ); \
            _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NHWC>( \
                { N, S, C }, x, mu, rsig, gamma, ds, db, dy, dx \
            ); \
        } \
    } \
    template <> void BatchNormBackwardInference<Tx, Tp, CPUContext>( \
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
        CPUContext*                 ctx) { \
        if (data_format == "NCHW") { \
            if (dgamma != nullptr) { \
                math::Set(C, Tp(0), dgamma, ctx); \
                math::Set(C, Tp(0), dbeta, ctx); \
                _BatchNormWGrad<Tx, Tp, StorageOrder::NCHW>( \
                    { N, C, S }, x, mu, rsig, dy, dgamma, dbeta \
                ); \
            } \
            _BatchNormInferenceGrad<Tx, Tp, StorageOrder::NCHW>( \
                N, C, S, rsig, gamma, dy, dx \
            ); \
        } else if (data_format == "NHWC") { \
            if (dgamma != nullptr) { \
                math::Set(C, Tp(0), dgamma, ctx); \
                math::Set(C, Tp(0), dbeta, ctx); \
                _BatchNormWGrad<Tx, Tp, StorageOrder::NHWC>( \
                    { N, S, C }, x, mu, rsig, dy, dgamma, dbeta \
                ); \
            } \
            _BatchNormInferenceGrad<Tx, Tp, StorageOrder::NHWC>( \
                N, C, S, rsig, gamma, dy, dx \
            ); \
        } \
    }

DEFINE_BACKWARD_KERNEL_LAUNCHER(float, float);
#undef DEFINE_BACKWARD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namespace dragon