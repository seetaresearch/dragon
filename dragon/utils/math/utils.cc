#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace utils {

bool IsRowwiseBroadcast(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    int* rows,
    int* cols,
    int* broadcast_1st) {
  auto a_size = std::accumulate(
      A_dims.data(),
      A_dims.data() + A_dims.size(),
      1,
      std::multiplies<int64_t>());
  auto b_size = std::accumulate(
      B_dims.data(),
      B_dims.data() + B_dims.size(),
      1,
      std::multiplies<int64_t>());
  if (broadcast_1st != nullptr) *broadcast_1st = 0;
  if (a_size == b_size) return false;

  vec64_t a_dims(b_size > a_size ? B_dims : A_dims);
  vec64_t b_dims(b_size > a_size ? A_dims : B_dims);

  auto num_dims = (int)std::max(a_dims.size(), b_dims.size());

  for (int i = (int)a_dims.size(); i < num_dims; ++i)
    a_dims.insert(a_dims.begin(), 1);
  for (int i = (int)b_dims.size(); i < num_dims; ++i)
    b_dims.insert(b_dims.begin(), 1);

  int pivot = num_dims - 1;
  *cols = 1;
  *rows = 1;
  for (; pivot >= 0 && a_dims[pivot] == b_dims[pivot]; --pivot) {
    *cols *= a_dims[pivot];
  }
  for (int i = pivot; i >= 0; --i) {
    if (b_dims[i] != 1) return false;
    *rows *= a_dims[i];
  }
  if (broadcast_1st != nullptr) {
    *broadcast_1st = a_size < b_size ? 1 : 0;
  }
  return true;
}

bool IsColwiseBroadcast(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    int* rows,
    int* cols,
    int* broadcast_1st) {
  auto a_size = std::accumulate(
      A_dims.data(),
      A_dims.data() + A_dims.size(),
      1,
      std::multiplies<int64_t>());
  auto b_size = std::accumulate(
      B_dims.data(),
      B_dims.data() + B_dims.size(),
      1,
      std::multiplies<int64_t>());
  if (broadcast_1st != nullptr) *broadcast_1st = 0;
  if (a_size == b_size) return false;

  vec64_t a_dims(b_size > a_size ? B_dims : A_dims);
  vec64_t b_dims(b_size > a_size ? A_dims : B_dims);

  auto num_dims = (int)std::max(a_dims.size(), b_dims.size());
  for (int i = (int)a_dims.size(); i < num_dims; ++i)
    a_dims.push_back(1);
  for (int i = (int)b_dims.size(); i < num_dims; ++i)
    b_dims.push_back(1);

  int pivot = 0;
  *cols = 1;
  *rows = 1;
  for (; pivot < num_dims && a_dims[pivot] == b_dims[pivot]; ++pivot) {
    *rows *= a_dims[pivot];
  }
  for (int i = pivot; i < num_dims; ++i) {
    if (b_dims[i] != 1) return false;
    *cols *= a_dims[i];
  }
  if (broadcast_1st != nullptr) {
    *broadcast_1st = a_size < b_size ? 1 : 0;
  }
  return true;
}

bool IsBinaryBroadcast(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    vec64_t& Y_dims) {
  Y_dims.resize(std::max(A_dims.size(), B_dims.size()));
  int i = A_dims.size() - 1;
  int j = B_dims.size() - 1;
  int k = Y_dims.size() - 1;
  for (; i >= 0 && j >= 0; --k) {
    const int A_dim = A_dims[i];
    const int B_dim = B_dims[j];
    if (A_dim == B_dim || A_dim == 1 || B_dim == 1) {
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

bool IsRowwiseReduce(
    const int num_dims,
    const int* X_dims,
    const int* Y_dims,
    int* rows,
    int* cols) {
  *rows = 1;
  int pivot = 0;
  for (; pivot < num_dims && Y_dims[pivot] == 1; ++pivot) {
    *rows *= X_dims[pivot];
  }
  *cols = 1;
  for (int i = pivot; i < num_dims; ++i) {
    if (X_dims[i] != Y_dims[i]) {
      return false;
    }
    *cols *= X_dims[i];
  }
  return true;
}

bool IsColwiseReduce(
    const int num_dims,
    const int* X_dims,
    const int* Y_dims,
    int* rows,
    int* cols) {
  *cols = 1;
  int pivot = num_dims - 1;
  for (; pivot >= 0 && Y_dims[pivot] == 1; --pivot) {
    *cols *= X_dims[pivot];
  }
  *rows = 1;
  for (int i = pivot; i >= 0; --i) {
    if (X_dims[i] != Y_dims[i]) {
      return false;
    }
    *rows *= X_dims[i];
  }
  return true;
}

void ComputeBinaryBroadcastDims(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    vec64_t& A_broadcast_dims,
    vec64_t& B_broadcast_dims,
    int64_t* C_broadcast_dims) {
  auto num_dims = std::max(A_dims.size(), B_dims.size());
  A_broadcast_dims.resize(num_dims);
  B_broadcast_dims.resize(num_dims);
  std::fill(
      A_broadcast_dims.begin(),
      A_broadcast_dims.begin() + num_dims - A_dims.size(),
      1);
  std::fill(
      B_broadcast_dims.begin(),
      B_broadcast_dims.begin() + num_dims - B_dims.size(),
      1);
  std::copy(
      A_dims.begin(),
      A_dims.end(),
      A_broadcast_dims.begin() + num_dims - A_dims.size());
  std::copy(
      B_dims.begin(),
      B_dims.end(),
      B_broadcast_dims.begin() + num_dims - B_dims.size());
  if (C_broadcast_dims != nullptr) {
    for (int i = 0; i < num_dims; ++i) {
      if (A_broadcast_dims[i] == 0 || B_broadcast_dims[i] == 0) {
        C_broadcast_dims[i] = 0;
      } else {
        C_broadcast_dims[i] =
            std::max(A_broadcast_dims[i], B_broadcast_dims[i]);
      }
    }
  }
}

void ComputeBinaryBroadcastStrides(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    vec64_t& A_broadcast_strides,
    vec64_t& B_broadcast_strides,
    vec64_t& Y_dims) {
  vec64_t A_broadcast_dims, B_broadcast_dims;
  ComputeBinaryBroadcastDims(
      A_dims, B_dims, A_broadcast_dims, B_broadcast_dims);
  auto num_dims = std::max(A_dims.size(), B_dims.size());
  A_broadcast_strides.resize(num_dims);
  B_broadcast_strides.resize(num_dims);
  Y_dims.resize(num_dims);
  int64_t A_stride = 1;
  int64_t B_stride = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    A_broadcast_strides[i] = A_broadcast_dims[i] == 1 ? 0 : A_stride;
    B_broadcast_strides[i] = B_broadcast_dims[i] == 1 ? 0 : B_stride;
    Y_dims[i] = std::max(A_broadcast_dims[i], B_broadcast_dims[i]);
    A_stride *= A_broadcast_dims[i];
    B_stride *= B_broadcast_dims[i];
  }
}

void ComputeBinaryBroadcastAxes(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    const vec64_t& Y_dims,
    vec32_t& A_broadcast_axes,
    vec32_t& B_broadcast_axes) {
  vec64_t A_broadcast_dims, B_broadcast_dims;
  ComputeBinaryBroadcastDims(
      A_dims, B_dims, A_broadcast_dims, B_broadcast_dims);
  auto num_dims = std::max(A_dims.size(), B_dims.size());
  CHECK_EQ(Y_dims.size(), num_dims);
  A_broadcast_axes.clear();
  B_broadcast_axes.clear();
  for (int i = 0; i < num_dims; ++i) {
    if (A_broadcast_dims[i] != Y_dims[i]) {
      CHECK_EQ(A_broadcast_dims[i], 1);
      A_broadcast_axes.push_back(i);
    }
    if (B_broadcast_dims[i] != Y_dims[i]) {
      CHECK_EQ(B_broadcast_dims[i], 1);
      B_broadcast_axes.push_back(i);
    }
  }
}

void TransposeAxesForReduce(
    const int num_dims,
    const int num_axes,
    const int* reduce_axes,
    int* transpose_axes) {
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

} // namespace utils

} // namespace math

} // namespace dragon
