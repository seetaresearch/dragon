/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * Codes are based on:
 *
 *  <https://github.com/pytorch/pytorch/blob/master/caffe2/utils/eigen_utils.h>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_EIGEN_UTILS_H_
#define DRAGON_UTILS_EIGEN_UTILS_H_

#include <Eigen/Core>

namespace dragon {

using EigenInnerStride = Eigen::InnerStride<Eigen::Dynamic>;
using EigenOuterStride = Eigen::OuterStride<Eigen::Dynamic>;

template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using EigenStridedVectorMap =
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, EigenInnerStride>;

template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenStridedVectorMap =
    Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, EigenInnerStride>;

template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

} // namespace dragon

#endif // DRAGON_UTILS_EIGEN_UTILS_H_
