#include "dragon/operators/vision/conv_op_base.h"
#include "dragon/core/workspace.h"
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
void ConvOpBase<Context>::GetBaseArguments() {
  auto kshape = OP_REPEATED_ARG(int64_t, "kernel_shape");
  auto dilations = OP_REPEATED_ARG(int64_t, "dilations");
  auto strides = OP_REPEATED_ARG(int64_t, "strides");
  auto pads = OP_REPEATED_ARG(int64_t, "pads");

  // Infer the number of spatial axes from the kernel shape
  num_axes_ = (int64_t)kshape.size();
  CHECK_GT(num_axes_, 0) << "\nInvalid size of <kernel_shape>.";

  for (int i = 0; i < num_axes_; ++i) {
    kshape_.push_back(i < kshape.size() ? kshape[i] : kshape[0]);
    dilations_.push_back(i < dilations.size() ? dilations[i] : dilations[0]);
    strides_.push_back(i < strides.size() ? strides[i] : strides[0]);
    pads_begin_.push_back(i < pads.size() ? pads[i] : pads[0]);
  }

  if ((int64_t)pads.size() == (num_axes_ * 2)) {
    for (int i = 0; i < num_axes_; ++i) {
      pads_end_.push_back(pads[num_axes_ + i]);
    }
  } else {
    pads_end_.assign(pads_begin_.begin(), pads_begin_.end());
  }

  bool skip_flag = true;
  for (int i = 0; i < num_axes_; ++i) {
    skip_flag &= (kshape_[i] == 1 && strides_[i] == 1);
    skip_flag &= (pads_begin_[i] == 0 && pads_end_[i] == 0);
    if (!skip_flag) break;
  }
  skip_im2col_ = skip_flag ? 1 : 0;
}

template <class Context>
void ConvOpBase<Context>::ComputeOutShape() {
  out_shape_.clear();
  vec64_t X_dims = Input(0).dims();
  int64_t in_size, out_size, k_size, pad_size;
  if (!Transposed()) {
    for (int i = 0; i < num_axes_; ++i) {
      in_size = X_dims[axis_ + i];
      k_size = dilations_[i] * (kshape_[i] - 1) + 1;
      if (!str::find(padding_, "SAME")) { // Explicit pads
        pad_size = pads_begin_[i] + pads_end_[i];
        out_size = (in_size + pad_size - k_size) / strides_[i] + 1;
      } else { // Auto pads
        out_size = (in_size + strides_[i] - 1) / strides_[i];
        pad_size = (out_size - 1) * strides_[i] + k_size - in_size;
        pad_size = std::max(pad_size, int64_t(0));
        DETERMINE_SAME_PADDING(pads_begin_, pads_end_);
      }
      out_shape_.push_back(out_size);
    }
  } else {
    int num_output_padding;
    output_padding(0, &num_output_padding);
    CHECK(num_output_padding == 0 || num_output_padding == num_axes_)
        << "\nExcepted 0 or " << num_axes_ << " ints for <output_padding>.";
    if (!str::find(padding_, "SAME")) { // Explicit pads
      for (int i = 0; i < num_axes_; ++i) {
        in_size = X_dims[axis_ + i];
        k_size = dilations_[i] * (kshape_[i] - 1) + 1;
        pad_size = pads_begin_[i] + pads_end_[i];
        out_size = strides_[i] * (in_size - 1) + k_size - pad_size;
        if (num_output_padding > 0) out_size += output_padding(i);
        out_shape_.push_back(out_size);
      }
    } else {
      // Auto pads
      int num_output_shape;
      output_shape(0, &num_output_shape);
      CHECK(num_output_shape == num_axes_)
          << "\nExcepted " << num_axes_ << " ints for <output_shape>.";
      for (int i = 0; i < num_axes_; ++i) {
        in_size = X_dims[axis_ + i];
        k_size = dilations_[i] * (kshape_[i] - 1) + 1;
        out_size = output_shape(i);
        pad_size = strides_[i] * (in_size - 1) + k_size;
        if (num_output_padding > 0) pad_size += output_padding(i);
        CHECK_GE(pad_size, out_size)
            << "\nThe output shape is incorrect."
            << "\nDimension of spatial axis " << i << " should be at most "
            << pad_size << ".";
        pad_size = strides_[i] * (in_size - 1) + k_size - out_size;
        pad_size = std::max(pad_size, int64_t(0));
        DETERMINE_SAME_PADDING(pads_begin_, pads_end_);
        out_shape_.push_back(out_size);
      }
    }
  }
}

template <class Context>
void ConvOpBase<Context>::Reshape(bool backward) {
  const auto& X = Input(0);
  const auto& W = Input(1);
  auto* Y_ref = backward ? &Input(-1) : Output(0);

  // Compute input and output channels.
  in_channels_ = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  if (out_channels_ <= 0) {
    // Infer output channels from the weights shape.
    out_channels_ = W.count() / (in_channels_ / group_);
    for (int i = 0; i < num_axes_; ++i) {
      out_channels_ /= kshape_[i];
    }
    CHECK_GT(out_channels_, 0) << "\nFailed to infer the out channels "
                               << "from weights: " << W.DimString();
  }
  if (Transposed()) {
    conv_out_channels_ = in_channels_;
    conv_in_channels_ = out_channels_;
  } else {
    conv_out_channels_ = out_channels_;
    conv_in_channels_ = in_channels_;
  }

  // Compute weight and bias shape in NCHW format.
  w_shape_ = {conv_out_channels_, conv_in_channels_ / group_};
  for (int i = 0; i < num_axes_; ++i) {
    w_shape_.push_back(kshape_[i]);
  }
  b_shape_ = {out_channels_};

  // Compute output shape.
  ComputeOutShape();
  if (backward) {
    if (Output(0)->has_name()) Output(0)->ReshapeLike(X);
    if (Output(1)->has_name()) Output(1)->ReshapeLike(W);
    if (Output(2)->has_name()) Output(2)->Reshape({out_channels_});
  } else {
    vec64_t Y_dims{X.dim(0)};
    if (data_format() == "NCHW") {
      Y_dims.push_back(out_channels_);
      for (int i = 0; i < num_axes_; ++i) {
        Y_dims.push_back(out_shape_[i]);
      }
    } else if (data_format() == "NHWC") {
      for (int i = 0; i < num_axes_; ++i) {
        Y_dims.push_back(out_shape_[i]);
      }
      Y_dims.push_back(out_channels_);
    }
    Output(0)->Reshape(Y_dims);
  }

  // Compute output dim.
  auto end_axis = X.ndim() - 1;
  if (data_format() == "NCHW") {
    if (Transposed()) {
      conv_out_dim_ = X.count(axis_);
    } else {
      conv_out_dim_ = Y_ref->count(axis_);
    }
    out_dim_ = Y_ref->count(axis_);
  } else if (data_format() == "NHWC") {
    if (Transposed()) {
      conv_out_dim_ = X.count(axis_, end_axis);
    } else {
      conv_out_dim_ = Y_ref->count(axis_, end_axis);
    }
    out_dim_ = Y_ref->count(axis_, end_axis);
  }

  // Compute strides.
  X_stride_ = X.stride(0);
  Y_stride_ = Y_ref->stride(0);
  kernel_dim_ = conv_in_channels_ / group_;
  for (int i = 0; i < num_axes_; ++i) {
    kernel_dim_ *= kshape_[i];
  }
  col_stride_ = kernel_dim_ * conv_out_dim_;
  W_stride_ = conv_out_channels_ * kernel_dim_ / group_;
  Y_stride1_ = conv_out_channels_ * conv_out_dim_ / group_;

  // Compute Im2Col arguments.
  in_shape_.clear();
  for (int i = 0; i < num_axes_; ++i) {
    if (Transposed()) {
      in_shape_.push_back(Y_ref->dim(axis_ + i));
      out_shape_[i] = X.dim(axis_ + i);
    } else {
      in_shape_.push_back(X.dim(axis_ + i));
    }
  }
  col_dim_ = kernel_dim_ * group_;
  for (int i = 0; i < num_axes_; ++i) {
    col_dim_ *= out_shape_[i];
  }
}

template class ConvOpBase<CPUContext>;
#ifdef USE_CUDA
template class ConvOpBase<CUDAContext>;
#endif
#ifdef USE_MPS
template class ConvOpBase<MPSContext>;
#endif
#ifdef USE_MLU
template class ConvOpBase<MLUContext>;
#endif

DEFINE_OP_REPEATED_ARG(int64_t, ConvOpBase, output_shape);
DEFINE_OP_REPEATED_ARG(int64_t, ConvOpBase, output_padding);

#undef DETERMINE_SAME_PADDING

} // namespace dragon
