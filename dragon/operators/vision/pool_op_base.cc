#include "dragon/operators/vision/pool_op_base.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

#define DETERMINE_SAME_PADDING(begins, ends) \
  if (padding_ != "SAME_UPPER") {            \
    ends[i] = pad_size >> 1;                 \
    begins[i] = pad_size - ends[i];          \
  } else {                                   \
    begins[i] = pad_size >> 1;               \
    ends[i] = pad_size - begins[i];          \
  }

template <class Context>
void PoolOpBase<Context>::GetBaseArguments() {
  auto kshape = OP_REPEATED_ARG(int64_t, "kernel_shape");
  auto strides = OP_REPEATED_ARG(int64_t, "strides");
  auto pads = OP_REPEATED_ARG(int64_t, "pads");

  // Infer the number of spatial axes from the kernel shape.
  num_axes_ = (int64_t)kshape.size();
  CHECK_GT(num_axes_, 0) << "\nInvalid size of <kernel_shape>.";

  for (int i = 0; i < num_axes_; i++) {
    if (global_pool_ > 0) {
      kshape_.push_back(-1);
      strides_.push_back(1);
      pads_begin_.push_back(0);
    } else {
      kshape_.push_back(i < kshape.size() ? kshape[i] : kshape[0]);
      strides_.push_back(i < strides.size() ? strides[i] : strides[0]);
      pads_begin_.push_back(i < pads.size() ? pads[i] : pads[0]);
    }
  }
  if (pads.size() == (size_t)(2 * num_axes_) && global_pool_ == 0) {
    pads_end_.assign(pads.begin() + num_axes_, pads.end());
  } else {
    pads_end_.assign(pads_begin_.begin(), pads_begin_.end());
  }
}

template <class Context>
void PoolOpBase<Context>::ComputeOutShape() {
  // Align input dimensions.
  in_dims_ = Input(0).dims();
  if (data_format() == "NHWC") {
    in_dims_.insert(in_dims_.begin() + 1, in_dims_.back());
    in_dims_.pop_back(); // Store dimensions in NCHW order.
  }

  // Adjust kernel shape for global pooling.
  if (global_pool_ > 0) {
    for (int i = 0; i < num_axes_; i++) {
      kshape_[i] = in_dims_[i + 2];
    }
  }

  // Compute output dimensions.
  auto floor_or_ceil = ceil_mode_ > 0
      ? static_cast<float (*)(float)>(&std::ceil)
      : static_cast<float (*)(float)>(&std::floor);
  out_dims_ = in_dims_;
  out_shape_ = Input(0).dims();
  int64_t in_size, k_size, pad_size;
  for (int i = 0; i < num_axes_; i++) {
    float out_size;
    in_size = in_dims_[i + 2], k_size = kshape_[i];
    if (!str::find(padding_, "SAME")) { // Explicit pads.
      pad_size = pads_begin_[i] + pads_end_[i];
      out_size = float(in_size + pad_size - k_size) / float(strides_[i]) + 1.f;
      out_size = floor_or_ceil(out_size);
    } else { // Auto pads.
      out_size = std::ceil(float(in_size) / float(strides_[i]));
      pad_size = ((int64_t)out_size - 1) * strides_[i] + k_size - in_size;
      pad_size = std::max(pad_size, int64_t(0));
      DETERMINE_SAME_PADDING(pads_begin_, pads_end_);
    }
    out_shape_[i + axis_] = out_dims_[i + 2] = out_size;
  }
}

template class PoolOpBase<CPUContext>;
#ifdef USE_CUDA
template class PoolOpBase<CUDAContext>;
#endif
#ifdef USE_MPS
template class PoolOpBase<MPSContext>;
#endif

#undef DETERMINE_SAME_PADDING

} // namespace dragon
