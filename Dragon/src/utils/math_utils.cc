#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

#include "utils/math_utils.h"

namespace dragon {

namespace utils {

void IncreaseIndexInDims(
    const int               ndims,
    const int*              dims,
    int*                    index) {
    for (int i = ndims - 1; i >= 0; --i) {
        ++index[i];
        if (index[i] >= dims[i]) {
            index[i] -= dims[i];
        } else {
            break;
        }
    }
}

bool IsRowwiseBroadcast(
    const std::vector<int64_t>&     A_dims,
    const std::vector<int64_t>&     B_dims,
    int*                            rows,
    int*                            cols) {
    auto a_size = std::accumulate(A_dims.data(),
        A_dims.data() + A_dims.size(), 1, std::multiplies<int64_t>());
    auto b_size = std::accumulate(B_dims.data(),
        B_dims.data() + B_dims.size(), 1, std::multiplies<int64_t>());
    if (a_size == b_size) return false;

    std::vector<int64_t> a_dims(b_size > a_size ? B_dims : A_dims);
    std::vector<int64_t> b_dims(b_size > a_size ? A_dims : B_dims);

    auto ndims = (int)std::max(a_dims.size(), b_dims.size());

    for (int i = (int)a_dims.size(); i < ndims; ++i) a_dims.insert(a_dims.begin(), 1);
    for (int i = (int)b_dims.size(); i < ndims; ++i) b_dims.insert(b_dims.begin(), 1);

    int pivot = ndims - 1; *cols = 1; *rows = 1;
    for (; pivot >= 0 && a_dims[pivot] == b_dims[pivot]; --pivot) {
        *cols *= a_dims[pivot];
    }
    for (int i = pivot; i >= 0; --i) {
        if (b_dims[i] != 1) return false;
        *rows *= a_dims[i];
    }
    return true;
}

bool IsColwiseBroadcast(
    const std::vector<int64_t>&     A_dims,
    const std::vector<int64_t>&     B_dims,
    int*                            rows,
    int*                            cols) {
    auto a_size = std::accumulate(A_dims.data(),
        A_dims.data() + A_dims.size(), 1, std::multiplies<int64_t>());
    auto b_size = std::accumulate(B_dims.data(),
        B_dims.data() + B_dims.size(), 1, std::multiplies<int64_t>());
    if (a_size == b_size) return false;

    std::vector<int64_t> a_dims(b_size > a_size ? B_dims : A_dims);
    std::vector<int64_t> b_dims(b_size > a_size ? A_dims : B_dims);

    auto ndims = (int)std::max(a_dims.size(), b_dims.size());
    for (int i = (int)a_dims.size(); i < ndims; ++i) a_dims.emplace_back(1);
    for (int i = (int)b_dims.size(); i < ndims; ++i) b_dims.emplace_back(1);

    int pivot = 0; *cols = 1; *rows = 1;
    for (; pivot < ndims && a_dims[pivot] == b_dims[pivot]; ++pivot) {
        *rows *= a_dims[pivot];
    }
    for (int i = pivot; i < ndims; ++i) {
        if (b_dims[i] != 1) return false;
        *cols *= a_dims[i];
    }
    return true;
}

bool IsRowwiseReduce(
    const int               ndims,
    const int*              A_dims,
    const int*              B_dims,
    int*                    rows,
    int*                    cols) {
    *rows = 1;
    int pivot = 0;
    for (; pivot < ndims && B_dims[pivot] == 1; ++pivot) {
        *rows *= A_dims[pivot];
    }
    *cols = 1;
    for (int i = pivot; i < ndims; ++i) {
        if (A_dims[i] != B_dims[i]) return false;
        *cols *= A_dims[i];
    }
    return true;
}

bool IsColwiseReduce(
    const int               ndims,
    const int*              A_dims,
    const int*              B_dims,
    int*                    rows,
    int*                    cols) {
    *cols = 1;
    int pivot = ndims - 1;
    for (; pivot >= 0 && B_dims[pivot] == 1; --pivot) {
        *cols *= A_dims[pivot];
    }
    *rows = 1;
    for (int i = pivot; i >= 0; --i) {
        if (A_dims[i] != B_dims[i]) return false;
        *rows *= A_dims[i];
    }
    return true;
}

void ComputeTransposedAxesForReduce(
    const int               ndimss,
    const int               naxes,
    const int*              reduce_axes,
    int*                    transpose_axes) {
    const int d = ndimss - naxes;
    std::copy_n(reduce_axes, naxes, transpose_axes + d);
    std::sort(transpose_axes + d, transpose_axes + ndimss);
    int p = 0, q = d;
    for (int i = 0; i < ndimss; ++i) {
        if (q < ndimss && i == transpose_axes[q]) ++q;
        else transpose_axes[p++] = i;
    }
}

void ComputeTransposedStrides(
    const int               ndimss,
    const int*              dims,
    const int*              axes,
    int*                    strides) {
    std::vector<int> buff(ndimss);
    int cur_stride = 1;
    for (int i = ndimss - 1; i >= 0; --i) {
        buff[i] = cur_stride;
        cur_stride *= dims[i];
    }
    for (int i = 0; i < ndimss; ++i) {
        strides[i] = buff[axes[i]];
    }
}

}  // namespace utils

}  // namespace dragon