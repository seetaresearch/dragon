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

#ifndef DRAGON_UTILS_MATH_BROADCAST_H_
#define DRAGON_UTILS_MATH_BROADCAST_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace math {

/*
 * Broadcast Functions.
 */

template <typename T, class Context>
DRAGON_API void Set(
    const int x_ndim,
    const int64_t* x_dims,
    const int y_ndim,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Add(
    const int a_ndim,
    const int64_t* a_dims,
    const int b_ndim,
    const int64_t* b_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Sub(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Mul(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Div(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void BitwiseAnd(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void BitwiseOr(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void BitwiseXor(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Pow(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Atan2(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Minimum(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Maximum(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void And(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Or(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Xor(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Equal(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void NotEqual(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Less(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void LessEqual(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Greater(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void GreaterEqual(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Where(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const int C_ndim,
    const int64_t* C_dims,
    const T* a,
    const T* b,
    const bool* c,
    T* y,
    Context* ctx);

/*
 * Broadcast Utilities.
 */

namespace utils {

template <typename DimT>
bool IsBinaryBroadcast(
    const vector<DimT>& A_dims,
    const vector<DimT>& B_dims,
    vector<DimT>& Y_dims) {
  Y_dims.resize(std::max(A_dims.size(), B_dims.size()));
  int i = A_dims.size() - 1;
  int j = B_dims.size() - 1;
  int k = Y_dims.size() - 1;
  for (; i >= 0 && j >= 0; --k) {
    const DimT A_dim = A_dims[i];
    const DimT B_dim = B_dims[j];
    if (A_dim == B_dim || A_dim == DimT(1) || B_dim == DimT(1)) {
      Y_dims[k] = std::max(A_dim, B_dim);
    } else {
      return false;
    }
    --i;
    --j;
  }
  for (; i >= 0; --i) {
    Y_dims[k--] = A_dims[i];
  }
  for (; j >= 0; --j) {
    Y_dims[k--] = B_dims[j];
  }
  return true;
}

template <typename DimT>
bool IsRowwiseBroadcast(
    const vector<DimT>& A_dims,
    const vector<DimT>& B_dims,
    DimT* rows,
    DimT* cols,
    DimT* broadcast_1st = nullptr) {
  auto a_size = std::accumulate(
      A_dims.data(),
      A_dims.data() + A_dims.size(),
      DimT(1),
      std::multiplies<DimT>());
  auto b_size = std::accumulate(
      B_dims.data(),
      B_dims.data() + B_dims.size(),
      DimT(1),
      std::multiplies<DimT>());
  if (broadcast_1st != nullptr) *broadcast_1st = 0;
  if (a_size == b_size) return false;
  vector<DimT> a_dims(b_size > a_size ? B_dims : A_dims);
  vector<DimT> b_dims(b_size > a_size ? A_dims : B_dims);
  int num_dims = std::max(a_dims.size(), b_dims.size());
  for (int i = (int)a_dims.size(); i < num_dims; ++i) {
    a_dims.insert(a_dims.begin(), 1);
  }
  for (int i = (int)b_dims.size(); i < num_dims; ++i) {
    b_dims.insert(b_dims.begin(), 1);
  }
  int pivot = num_dims - 1;
  *cols = *rows = DimT(1);
  for (; pivot >= 0 && a_dims[pivot] == b_dims[pivot]; --pivot) {
    *cols *= a_dims[pivot];
  }
  for (int i = pivot; i >= 0; --i) {
    if (b_dims[i] != DimT(1)) return false;
    *rows *= a_dims[i];
  }
  if (broadcast_1st != nullptr) {
    *broadcast_1st = a_size < b_size ? 1 : 0;
  }
  return true;
}

template <typename DimT>
bool IsColwiseBroadcast(
    const vector<DimT>& A_dims,
    const vector<DimT>& B_dims,
    DimT* rows,
    DimT* cols,
    DimT* broadcast_1st = nullptr) {
  auto a_size = std::accumulate(
      A_dims.data(),
      A_dims.data() + A_dims.size(),
      DimT(1),
      std::multiplies<int64_t>());
  auto b_size = std::accumulate(
      B_dims.data(),
      B_dims.data() + B_dims.size(),
      DimT(1),
      std::multiplies<DimT>());
  if (broadcast_1st != nullptr) *broadcast_1st = 0;
  if (a_size == b_size) return false;
  vector<DimT> a_dims(b_size > a_size ? B_dims : A_dims);
  vector<DimT> b_dims(b_size > a_size ? A_dims : B_dims);
  int num_dims = std::max(a_dims.size(), b_dims.size());
  for (int i = (int)a_dims.size(); i < num_dims; ++i) {
    a_dims.push_back(DimT(1));
  }
  for (int i = (int)b_dims.size(); i < num_dims; ++i) {
    b_dims.push_back(DimT(1));
  }
  int pivot = 0;
  *cols = *rows = DimT(1);
  for (; pivot < num_dims && a_dims[pivot] == b_dims[pivot]; ++pivot) {
    *rows *= a_dims[pivot];
  }
  for (int i = pivot; i < num_dims; ++i) {
    if (b_dims[i] != DimT(1)) return false;
    *cols *= a_dims[i];
  }
  if (broadcast_1st != nullptr) {
    *broadcast_1st = a_size < b_size ? 1 : 0;
  }
  return true;
}

template <typename DimT>
void ComputeBroadcastDims(
    const vector<DimT>& A_dims,
    const vector<DimT>& B_dims,
    vector<DimT>& A_broadcast_dims,
    vector<DimT>& B_broadcast_dims,
    DimT* C_broadcast_dims = nullptr) {
  int num_dims = std::max(A_dims.size(), B_dims.size());
  A_broadcast_dims.resize(num_dims);
  B_broadcast_dims.resize(num_dims);
  std::fill(
      A_broadcast_dims.begin(),
      A_broadcast_dims.begin() + num_dims - A_dims.size(),
      DimT(1));
  std::fill(
      B_broadcast_dims.begin(),
      B_broadcast_dims.begin() + num_dims - B_dims.size(),
      DimT(1));
  std::copy(
      A_dims.begin(),
      A_dims.end(),
      A_broadcast_dims.begin() + num_dims - A_dims.size());
  std::copy(
      B_dims.begin(),
      B_dims.end(),
      B_broadcast_dims.begin() + num_dims - B_dims.size());
  if (C_broadcast_dims == nullptr) return;
  for (int i = 0; i < num_dims; ++i) {
    if (A_broadcast_dims[i] == DimT(0) || B_broadcast_dims[i] == DimT(0)) {
      C_broadcast_dims[i] = DimT(0);
    } else {
      C_broadcast_dims[i] = std::max(A_broadcast_dims[i], B_broadcast_dims[i]);
    }
  }
}

template <typename DimT, typename StrideT>
void ComputeBroadcastStrides(
    const vector<DimT>& A_dims,
    const vector<DimT>& B_dims,
    vector<StrideT>& A_broadcast_strides,
    vector<StrideT>& B_broadcast_strides,
    vector<DimT>& Y_dims) {
  vector<DimT> A_broadcast_dims, B_broadcast_dims;
  ComputeBroadcastDims(A_dims, B_dims, A_broadcast_dims, B_broadcast_dims);
  int num_dims = std::max(A_dims.size(), B_dims.size());
  A_broadcast_strides.resize(num_dims);
  B_broadcast_strides.resize(num_dims);
  Y_dims.resize(num_dims);
  StrideT A_stride = StrideT(1);
  StrideT B_stride = StrideT(1);
  for (int i = num_dims - 1; i >= 0; --i) {
    A_broadcast_strides[i] =
        (A_broadcast_dims[i] == DimT(1) ? StrideT(0) : A_stride);
    B_broadcast_strides[i] =
        (B_broadcast_dims[i] == DimT(1) ? StrideT(0) : B_stride);
    Y_dims[i] = std::max(A_broadcast_dims[i], B_broadcast_dims[i]);
    A_stride *= A_broadcast_dims[i];
    B_stride *= B_broadcast_dims[i];
  }
}

template <typename DimT, typename AxisT>
void ComputeBroadcastAxes(
    const vector<DimT>& A_dims,
    const vector<DimT>& B_dims,
    const vector<DimT>& Y_dims,
    vector<AxisT>& A_broadcast_axes,
    vector<AxisT>& B_broadcast_axes) {
  vector<DimT> A_broadcast_dims, B_broadcast_dims;
  ComputeBroadcastDims(A_dims, B_dims, A_broadcast_dims, B_broadcast_dims);
  int num_dims = std::max(A_dims.size(), B_dims.size());
  CHECK_EQ(Y_dims.size(), num_dims);
  A_broadcast_axes.clear();
  B_broadcast_axes.clear();
  for (int i = 0; i < num_dims; ++i) {
    if (A_broadcast_dims[i] != Y_dims[i]) {
      CHECK_EQ(A_broadcast_dims[i], DimT(1));
      A_broadcast_axes.push_back(AxisT(i));
    }
    if (B_broadcast_dims[i] != Y_dims[i]) {
      CHECK_EQ(B_broadcast_dims[i], DimT(1));
      B_broadcast_axes.push_back(AxisT(i));
    }
  }
}

} // namespace utils

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_BROADCAST_H_
