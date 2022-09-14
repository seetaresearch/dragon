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

#ifndef DRAGON_UTILS_MATH_TRANSPOSE_H_
#define DRAGON_UTILS_MATH_TRANSPOSE_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace math {

/*
 * Transpose Functions.
 */

template <typename T, class Context>
DRAGON_API void Transpose(
    const int num_dims,
    const int64_t* dims,
    const int64_t* axes,
    const T* x,
    T* y,
    Context* ctx);

/*
 * Transpose Utilities.
 */

namespace utils {

template <typename DimT, typename StrideT>
void ComputeStrides(const int num_dims, const DimT* dims, StrideT* strides) {
  int64_t cur_stride = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    strides[i] = StrideT(cur_stride);
    cur_stride *= int64_t(dims[i]);
  }
}

template <typename DimT, typename AxisT, typename StrideT>
void ComputeTransposeStrides(
    const int num_dims,
    const DimT* dims,
    const AxisT* axes,
    StrideT* strides) {
  vec64_t buf(num_dims);
  int64_t cur_stride = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    buf[i] = cur_stride;
    cur_stride *= int64_t(dims[i]);
  }
  for (int i = 0; i < num_dims; ++i) {
    strides[i] = StrideT(buf[axes[i]]);
  }
}

template <typename DimT, typename AxisT>
void CollapseTransposeAxes(
    const int num_dims,
    const DimT* dims,
    const AxisT* axes,
    vector<DimT>& new_dims,
    vector<AxisT>& new_axes) {
  new_dims = vector<DimT>(dims, dims + num_dims);
  new_axes = vector<AxisT>({axes[0]});
  vector<AxisT> collapse_axes;
  for (int i = 1; i < num_dims; ++i) {
    if (axes[i] - 1 == axes[i - 1]) {
      collapse_axes.push_back(axes[i]);
      new_dims[axes[i]] *= new_dims[axes[i] - 1];
      new_dims[axes[i] - 1] = -1;
    } else {
      new_axes.push_back(axes[i]);
    }
  }
  const auto& erase_iter = std::remove_if(
      new_dims.begin(), new_dims.end(), [](int x) { return x == -1; });
  new_dims.erase(erase_iter, new_dims.end());
  for (int i = 0; i < new_axes.size(); ++i) {
    const auto axis = new_axes[i];
    for (auto collapse_axis : collapse_axes) {
      if (axis > collapse_axis) new_axes[i]--;
    }
  }
}

} // namespace utils

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_TRANSPOSE_H_
