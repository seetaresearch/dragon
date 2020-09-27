#include "dragon/operators/vision/conv_op_base.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

#define DETERMINE_SAME_PADDING(l, r) \
  if (padding_ != "SAME_UPPER") {    \
    r[i] = pad_size >> 1;            \
    l[i] = pad_size - r[i];          \
  } else {                           \
    l[i] = pad_size >> 1;            \
    r[i] = pad_size - l[i];          \
  }

template <class Context>
void ConvOpBase<Context>::ComputeOutShape() {
  out_shape_.clear();
  vec64_t X_dims = Input(0).dims();
  int64_t in_size, out_size, k_size, pad_size;
  if (!Transposed()) {
    for (int i = 0; i < num_axes_; i++) {
      in_size = X_dims[axis_ + i];
      k_size = dilation_[i] * (kshape_[i] - 1) + 1;
      if (!str::find(padding_, "SAME")) { // Explicit pads
        pad_size = pad_l_[i] + pad_r_[i];
        out_size = (in_size + pad_size - k_size) / stride_[i] + 1;
      } else { // Auto pads
        out_size = (in_size + stride_[i] - 1) / stride_[i];
        pad_size = (out_size - 1) * stride_[i] + k_size - in_size;
        pad_size = std::max(pad_size, int64_t(0));
        DETERMINE_SAME_PADDING(pad_l_, pad_r_);
      }
      out_shape_.push_back(out_size);
    }
  } else {
    int num_output_padding;
    output_padding(0, &num_output_padding);
    CHECK(num_output_padding == 0 || num_output_padding == num_axes_)
        << "\nExcepted 0 or " << num_axes_ << " ints for <output_padding>.";
    if (!str::find(padding_, "SAME")) { // Explicit pads
      for (int i = 0; i < num_axes_; i++) {
        in_size = X_dims[axis_ + i];
        k_size = dilation_[i] * (kshape_[i] - 1) + 1;
        pad_size = pad_l_[i] + pad_r_[i];
        out_size = stride_[i] * (in_size - 1) + k_size - pad_size;
        if (num_output_padding > 0) out_size += output_padding(i);
        out_shape_.push_back(out_size);
      }
    } else {
      // Auto pads
      int num_output_shape;
      output_shape(0, &num_output_shape);
      CHECK(num_output_shape == num_axes_)
          << "\nExcepted " << num_axes_ << " ints for <output_shape>.";
      for (int i = 0; i < num_axes_; i++) {
        in_size = X_dims[axis_ + i];
        k_size = dilation_[i] * (kshape_[i] - 1) + 1;
        out_size = output_shape(i);
        pad_size = stride_[i] * (in_size - 1) + k_size;
        if (num_output_padding > 0) pad_size += output_padding(i);
        CHECK_GE(pad_size, out_size)
            << "\nThe output shape is incorrect."
            << "\nDimension of spatial axis " << i << " should be at most "
            << pad_size << ".";
        pad_size = stride_[i] * (in_size - 1) + k_size - out_size;
        pad_size = std::max(pad_size, int64_t(0));
        DETERMINE_SAME_PADDING(pad_l_, pad_r_);
        out_shape_.push_back(out_size);
      }
    }
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::Wx(const T* x, const T* w, T* y, bool skip) {
  auto* col = x;
  if (!is_1x1_) {
    auto* scratch =
        ctx()->workspace()->template data<T, Context>({col_dim_})[0];
    if (!skip) Im2Col(x, scratch);
    col = scratch;
  }
  for (int g = 0; g < group_; g++) {
    if (data_format() == "NCHW") {
      math::Gemm(
          CblasNoTrans,
          CblasNoTrans,
          conv_out_channels_ / group_,
          conv_out_dim_,
          kernel_dim_,
          1.f,
          w + w_offset_ * g,
          col + col_offset_ * g,
          0.f,
          y + out_offset_ * g,
          ctx());
    } else if (data_format() == "NHWC") {
      math::Gemm(
          CblasNoTrans,
          CblasTrans,
          conv_out_dim_,
          conv_out_channels_,
          kernel_dim_,
          1.f,
          col,
          w,
          0.f,
          y,
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::Pb(const T* bias, T* y) {
  if (data_format() == "NCHW") {
    kernel::BiasAdd(
        Input(0).dim(0), out_channels_, out_dim_, y, bias, y, ctx());
  } else if (data_format() == "NHWC") {
    kernel::BiasAdd(
        Input(0).dim(0) * out_dim_, out_channels_, 1, y, bias, y, ctx());
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::Dx(const T* dy, const T* w, T* dx) {
  auto* col = is_1x1_
      ? dx
      : ctx()->workspace()->template data<T, Context>({col_dim_})[0];
  for (int g = 0; g < group_; g++) {
    if (data_format() == "NCHW") {
      math::Gemm(
          CblasTrans,
          CblasNoTrans,
          kernel_dim_,
          conv_out_dim_,
          conv_out_channels_ / group_,
          1.f,
          w + w_offset_ * g,
          dy + out_offset_ * g,
          0.f,
          col + col_offset_ * g,
          ctx());
    } else if (data_format() == "NHWC") {
      math::Gemm(
          CblasNoTrans,
          CblasNoTrans,
          conv_out_dim_,
          kernel_dim_,
          conv_out_channels_,
          1.f,
          dy,
          w,
          0.f,
          col,
          ctx());
    }
  }
  if (!is_1x1_) Col2Im(col, dx);
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::Dw(const T* dy, const T* x, T* dw, bool accum) {
  auto* col = x;
  if (!is_1x1_) {
    auto* scratch =
        ctx()->workspace()->template data<T, Context>({col_dim_})[0];
    Im2Col(x, scratch);
    col = scratch;
  }
  for (int g = 0; g < group_; g++) {
    if (data_format() == "NCHW") {
      math::Gemm(
          CblasNoTrans,
          CblasTrans,
          conv_out_channels_ / group_,
          kernel_dim_,
          conv_out_dim_,
          1.f,
          dy + out_offset_ * g,
          col + col_offset_ * g,
          accum ? 1.f : 0.f,
          dw + w_offset_ * g,
          ctx());
    } else if (data_format() == "NHWC") {
      math::Gemm(
          CblasTrans,
          CblasNoTrans,
          conv_out_channels_,
          kernel_dim_,
          conv_out_dim_,
          1.f,
          dy,
          col,
          accum ? 1.f : 0.f,
          dw,
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::Db(const T* dy, T* db) {
  vec32_t dims, axes;
  if (data_format() == "NCHW") {
    dims = {(int)Input(0).dim(0), (int)out_channels_, (int)out_dim_};
    axes = {0, 2};
  } else if (data_format() == "NHWC") {
    dims = {(int)Input(0).dim(0), (int)out_dim_, (int)out_channels_};
    axes = {0, 1};
  }
  math::ReduceSum(3, dims.data(), 2, axes.data(), 1.f, dy, db, ctx());
}

template <class Context>
void ConvOpBase<Context>::Setup(int num_axes) {
  num_axes_ = num_axes;
  auto pads = OP_REPEATED_ARG(int64_t, "pads");
  auto strides = OP_REPEATED_ARG(int64_t, "strides");
  auto kshape = OP_REPEATED_ARG(int64_t, "kernel_shape");
  auto dilations = OP_REPEATED_ARG(int64_t, "dilations");

  auto at = [&](const vec64_t& vec, int i) {
    return i < vec.size() ? vec[i] : vec[0];
  };

  for (int i = 0; i < num_axes; i++) {
    pad_l_.push_back(at(pads, i));
    stride_.push_back(at(strides, i));
    kshape_.push_back(at(kshape, i));
    dilation_.push_back(at(dilations, i));
  }

  if ((int64_t)pads.size() == (num_axes * 2)) {
    for (int i = 0; i < num_axes; i++) {
      pad_r_.push_back(pads[num_axes + i]);
    }
  } else {
    pad_r_.assign(pad_l_.begin(), pad_l_.end());
  }

  bool flag_1x1 = true;
  for (int i = 0; i < num_axes; i++) {
    flag_1x1 &=
        (pad_l_[i] == 0 && pad_r_[i] == 0 && stride_[i] == 1 &&
         kshape_[i] == 1);
    if (!flag_1x1) break;
  }
  is_1x1_ = flag_1x1 ? 1 : 0;
}

template <class Context>
void ConvOpBase<Context>::Reshape(bool backward) {
  const auto& X = Input(0);
  const auto& W = Input(1);
  auto* Y_ref = backward ? &Input(-1) : Output(0);

  // Determine the in/out channels
  in_channels_ = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  if (out_channels_ <= 0) {
    // Infer the out channels from the weights shape
    out_channels_ = W.count() / (in_channels_ / group_);
    for (int i = 0; i < num_axes_; i++) {
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

  // Determine the weight and bias shape
  // Weight shape is assumed as NCHW format
  // whatever to compute the fans correctly
  w_shape_ = {conv_out_channels_, conv_in_channels_ / group_};
  for (int i = 0; i < num_axes_; i++) {
    w_shape_.push_back(kshape_[i]);
  }
  b_shape_ = {out_channels_};

  // Determine the output shape
  ComputeOutShape();
  if (backward) {
    if (Output(0)->has_name()) Output(0)->ReshapeLike(X);
    if (Output(1)->has_name()) Output(1)->ReshapeLike(W);
    if (Output(2)->has_name()) Output(2)->Reshape({out_channels_});
  } else {
    vec64_t Y_dims{X.dim(0)};
    if (data_format() == "NCHW") {
      Y_dims.push_back(out_channels_);
      for (int i = 0; i < num_axes_; i++) {
        Y_dims.push_back(out_shape_[i]);
      }
    } else if (data_format() == "NHWC") {
      for (int i = 0; i < num_axes_; i++) {
        Y_dims.push_back(out_shape_[i]);
      }
      Y_dims.push_back(out_channels_);
    }
    Output(0)->Reshape(Y_dims);
  }

  // Determine the output dim
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

  // Compute the miscellaneous
  x_offset_ = X.stride(0);
  y_offset_ = Y_ref->stride(0);
  kernel_dim_ = conv_in_channels_ / group_;
  for (int i = 0; i < num_axes_; i++) {
    kernel_dim_ *= kshape_[i];
  }
  col_offset_ = kernel_dim_ * conv_out_dim_;
  w_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  out_offset_ = conv_out_channels_ * conv_out_dim_ / group_;

  // Compute the arguments for im2col/col2im
  in_shape_.clear();
  for (int i = 0; i < num_axes_; i++) {
    if (Transposed()) {
      in_shape_.push_back(Y_ref->dim(axis_ + i));
      out_shape_[i] = X.dim(axis_ + i);
    } else {
      in_shape_.push_back(X.dim(axis_ + i));
    }
  }
  col_dim_ = kernel_dim_ * group_;
  for (int i = 0; i < num_axes_; i++) {
    col_dim_ *= out_shape_[i];
  }
}

#define INSTANTIATE_API(Context, T)                                    \
  template void ConvOpBase<Context>::Wx(const T*, const T*, T*, bool); \
  template void ConvOpBase<Context>::Pb(const T*, T*);                 \
  template void ConvOpBase<Context>::Dx(const T*, const T*, T*);       \
  template void ConvOpBase<Context>::Dw(const T*, const T*, T*, bool); \
  template void ConvOpBase<Context>::Db(const T*, T*);

template class ConvOpBase<CPUContext>;
INSTANTIATE_API(CPUContext, float);
INSTANTIATE_API(CPUContext, double);

#ifdef USE_CUDA
template class ConvOpBase<CUDAContext>;
INSTANTIATE_API(CUDAContext, float);
INSTANTIATE_API(CUDAContext, double);
#endif

#undef INSTANTIATE_API
#undef DETERMINE_SAME_PADDING

} // namespace dragon
