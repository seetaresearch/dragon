#include "dragon/operators/vision/pool_op_base.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

#define SAME_PADDING(A, B)   \
  A[i] = padding_needed / 2; \
  B[i] = padding_needed - A[i]

template <class Context>
void PoolOpBase<Context>::Setup(int num_axes) {
  num_axes_ = num_axes;

  auto at = [&](const vec64_t& vec, int i) {
    return i < vec.size() ? vec[i] : vec[0];
  };
  auto pads = OpArgs<int64_t>("pads");
  auto strides = OpArgs<int64_t>("strides");
  auto kshape = OpArgs<int64_t>("kernel_shape");
  for (int i = 0; i < num_axes_; i++) {
    if (global_pool_) {
      pad_l_.push_back(0);
      stride_.push_back(1);
      kshape_.push_back(-1);
    } else {
      pad_l_.push_back(at(pads, i));
      stride_.push_back(at(strides, i));
      kshape_.push_back(at(kshape, i));
    }
  }
  if (pads.size() == (size_t)(2 * num_axes_)) {
    pad_r_.assign(pads.begin() + num_axes_, pads.end());
  } else {
    pad_r_.assign(pad_l_.begin(), pad_l_.end());
  }
}

template <class Context>
void PoolOpBase<Context>::ComputeOutShape() {
  // Determine the input dimensions
  in_dims_ = Input(0).dims();
  if (data_format() == "NHWC") {
    in_dims_.insert(in_dims_.begin() + 1, in_dims_.back());
    in_dims_.pop_back(); // Store dimensions in NCHW order
  }

  // Adjust the kernel shape for global pooling
  if (global_pool_ > 0) {
    for (int i = 0; i < num_axes_; i++)
      kshape_[i] = in_dims_[i + 2];
  }

  // Adjust the pads for SAME padding algorithm
  if (str::find(padding_, "SAME")) {
    for (int i = 0; i < num_axes_; i++) {
      auto idm = in_dims_[i + 2];
      int64_t odm = (idm + stride_[i] - 1) / (float)stride_[i];
      auto padding_needed =
          std::max((int64_t)0, (odm - 1) * stride_[i] + kshape_[i] - idm);
      if (padding_ == "SAME_UPPER") {
        SAME_PADDING(pad_l_, pad_r_);
      } else {
        SAME_PADDING(pad_r_, pad_l_);
      } /*! SAME_LOWER or SAME */
    }
  }

  // Compute the output dimensions
  auto floor_or_ceil = ceil_mode_ > 0
      ? static_cast<float (*)(float)>(&std::ceil)
      : static_cast<float (*)(float)>(&std::floor);

  out_dims_ = in_dims_;
  out_shape_ = Input(0).dims();

  for (int i = 0; i < num_axes_; i++) {
    auto in_dim = in_dims_[i + 2];
    if (!str::find(padding_, "SAME")) {
      // Explicit pads
      in_dim += pad_l_[i] + pad_r_[i];
      out_shape_[i + axis_] = out_dims_[i + 2] =
          floor_or_ceil((in_dim - kshape_[i]) / (float)stride_[i]) + 1;
    } else {
      // Auto pads
      out_shape_[i + axis_] = out_dims_[i + 2] =
          floor_or_ceil(in_dim / (float)stride_[i]);
    }
  }
}

template class PoolOpBase<CPUContext>;
#ifdef USE_CUDA
template class PoolOpBase<CUDAContext>;
#endif

#undef SAME_PADDING

} // namespace dragon
