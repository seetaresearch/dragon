/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_DEVICE_COMMON_EIGEN_H_
#define DRAGON_UTILS_DEVICE_COMMON_EIGEN_H_

#include <Eigen/Core>

namespace dragon {

using EigenInnerStride = Eigen::InnerStride<Eigen::Dynamic>;
using EigenOuterStride = Eigen::OuterStride<Eigen::Dynamic>;

/*
 * Vector
 */

template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using EigenStridedVectorMap =
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, EigenInnerStride>;

template <typename T>
using ConstEigenStridedVectorMap =
    Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, EigenInnerStride>;

/*
 * VectorArray
 */

template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
using EigenVectorArrayMap2 = Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>>;

template <typename T>
using ConstEigenVectorArrayMap2 =
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>>;

template <typename T>
using EigenStridedVectorArrayMap =
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>, 0, EigenInnerStride>;

template <typename T>
using ConstEigenStridedVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>, 0, EigenInnerStride>;

/*
 * Array
 */

template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

/*
 * Matrix
 */

template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

} // namespace dragon

#endif // DRAGON_UTILS_DEVICE_COMMON_EIGEN_H_
