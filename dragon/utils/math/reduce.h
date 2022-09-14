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

#ifndef DRAGON_UTILS_MATH_REDUCE_H_
#define DRAGON_UTILS_MATH_REDUCE_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace math {

/*
 * Reduce Functions.
 */

template <typename T, class Context>
DRAGON_API void ReduceMax(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const float scale,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void ReduceMin(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const float scale,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void ReduceSum(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const float scale,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void ReduceSumSqr(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const float scale,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void ReduceL1(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const float scale,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void
Sum(const int N, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API T Sum(const int N, const float alpha, const T* x, Context* ctx);

/*
 * Reduce Utilities.
 */

namespace utils {

template <typename DimT>
bool IsRowwiseReduce(
    const int num_dims,
    const DimT* X_dims,
    const DimT* Y_dims,
    DimT* rows,
    DimT* cols) {
  *rows = *cols = DimT(1);
  int pivot = 0;
  for (; pivot < num_dims && Y_dims[pivot] == 1; ++pivot) {
    *rows *= X_dims[pivot];
  }
  for (int i = pivot; i < num_dims; ++i) {
    if (X_dims[i] != Y_dims[i]) {
      return false;
    }
    *cols *= X_dims[i];
  }
  return true;
}

template <typename DimT>
bool IsColwiseReduce(
    const int num_dims,
    const DimT* X_dims,
    const DimT* Y_dims,
    DimT* rows,
    DimT* cols) {
  *cols = *rows = DimT(1);
  int pivot = num_dims - 1;
  for (; pivot >= 0 && Y_dims[pivot] == 1; --pivot) {
    *cols *= X_dims[pivot];
  }
  for (int i = pivot; i >= 0; --i) {
    if (X_dims[i] != Y_dims[i]) {
      return false;
    }
    *rows *= X_dims[i];
  }
  return true;
}

template <typename AxisT>
void TransposeAxesForReduce(
    const int num_dims,
    const int num_axes,
    const AxisT* reduce_axes,
    AxisT* transpose_axes) {
  const int d = num_dims - num_axes;
  std::copy_n(reduce_axes, num_axes, transpose_axes + d);
  std::sort(transpose_axes + d, transpose_axes + num_dims);
  int p = 0, q = d;
  for (int i = 0; i < num_dims; ++i) {
    if (q < num_dims && i == transpose_axes[q]) {
      ++q;
    } else {
      transpose_axes[p++] = i;
    }
  }
}

template <typename DimT, typename AxisT>
void CollapseReduceAxes(
    const int num_dims,
    const DimT* dims,
    const int num_axes,
    const AxisT* axes,
    vector<DimT>& new_dims,
    vector<AxisT>& new_axes) {
  new_dims.clear();
  new_axes.clear();
  DimT new_dim = DimT(0);
  AxisT new_axis = AxisT(0);
  auto reduce_dims = vector<DimT>(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    reduce_dims[axes[i]] = DimT(-1);
  }
  for (int i = 0; i < num_dims; ++i) {
    if (dims[i] == DimT(1)) continue;
    if (reduce_dims[i] == DimT(-1)) {
      if (new_axes.empty() || new_axis != new_axes.back()) {
        if (new_dim > DimT(0)) new_dims.push_back(new_dim);
        new_dim = dims[i];
        new_axes.push_back(new_axis);
      } else {
        new_dim *= dims[i];
      }
    } else {
      if (!new_axes.empty() && new_axis == new_axes.back()) {
        if (new_dim > DimT(0)) new_dims.push_back(new_dim);
        new_dim = DimT(0);
        new_axis += AxisT(1);
      }
      if (new_dim == DimT(0)) {
        new_dim = dims[i];
        new_axis += AxisT(1);
      } else {
        new_dim *= dims[i];
      }
    }
  }
  if (new_dim > DimT(0)) new_dims.push_back(new_dim);
}

} // namespace utils

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_REDUCE_H_
