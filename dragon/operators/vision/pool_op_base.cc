#include "dragon/operators/vision/pool_op_base.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

#define DETERMINE_SAME_PADDING(l, r) \
  if (padding_ != "SAME_UPPER") {    \
    l[i] = pad_size / 2;             \
    r[i] = pad_size - l[i];          \
  } else {                           \
    r[i] = pad_size / 2;             \
    l[i] = pad_size - r[i];          \
  }

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

  // Compute the output dimensions
  auto floor_or_ceil = ceil_mode_ > 0
      ? static_cast<float (*)(float)>(&std::ceil)
      : static_cast<float (*)(float)>(&std::floor);
  out_dims_ = in_dims_;
  out_shape_ = Input(0).dims();
  int64_t in_size, k_size, pad_size;
  for (int i = 0; i < num_axes_; i++) {
    float out_size;
    in_size = in_dims_[i + 2], k_size = kshape_[i];
    if (!str::find(padding_, "SAME")) { // Explicit pads
      pad_size = pad_l_[i] + pad_r_[i];
      out_size = float(in_size + pad_size - k_size) / float(stride_[i]) + 1.f;
      out_size = floor_or_ceil(out_size);
    } else { // Auto pads
      out_size = std::ceil(float(in_size) / float(stride_[i]));
      pad_size = ((int64_t)out_size - 1) * stride_[i] + k_size - in_size;
      pad_size = std::max(pad_size, int64_t(0));
      DETERMINE_SAME_PADDING(pad_l_, pad_r_);
    }
    out_shape_[i + axis_] = out_dims_[i + 2] = out_size;
  }
}

template class PoolOpBase<CPUContext>;
#ifdef USE_CUDA
template class PoolOpBase<CUDAContext>;
#endif

#undef DETERMINE_SAME_PADDING

} // namespace dragon
