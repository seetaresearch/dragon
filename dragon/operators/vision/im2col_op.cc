#include "dragon/operators/vision/im2col_op.h"
#include "dragon/core/workspace.h"
#include "dragon/operators/vision/conv_op_impl.h"

namespace dragon {

template <class Context>
template <typename T>
void Im2ColOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  ConvOpBase<Context>::ComputeOutShape();
  if (str::find(this->padding_, "SAME")) SET_INPUT_SPEC(0);

  vec64_t Y_dims(X.dims());
  auto im_channels_iter = Y_dims.begin() + 1;
  if (data_format() == "NHWC") im_channels_iter += num_axes_;
  conv_in_channels_ = *im_channels_iter;
  in_shape_.clear();
  for (int i = 0; i < num_axes_; ++i) {
    in_shape_.push_back(X.dim(axis_ + i));
    Y_dims[axis_ + i] = out_shape_[i];
    *im_channels_iter *= kshape_[i];
  }

  auto* x = X.template data<T, Context>();
  auto* y = Y->Reshape(Y_dims)->template mutable_data<T, Context>();
  for (int i = 0; i < X.dim(0); ++i) {
    this->Im2Col(x + i * X.stride(0), y + i * Y->stride(0));
  }
}

template <class Context>
template <typename T>
void Col2ImOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  if (str::find(this->padding_, "SAME")) {
    auto* Y_ref = workspace()->TryGetTensor(handle() + "/X_spec:0");
    if (Y_ref != nullptr) {
      // Get output shape from the spec if computed.
      this->output_shape_.resize(num_axes_);
      for (int i = 0; i < num_axes_; ++i) {
        this->output_shape_[i] = Y_ref->dim(axis_ + i);
      }
    }
  }
  ConvOpBase<Context>::ComputeOutShape();

  vec64_t Y_dims(X.dims());
  auto im_channels_iter = Y_dims.begin() + 1;
  if (data_format() == "NHWC") im_channels_iter += num_axes_;
  in_shape_.clear();
  for (int i = 0; i < num_axes_; ++i) {
    in_shape_.push_back(X.dim(axis_ + i));
    Y_dims[axis_ + i] = out_shape_[i];
    *im_channels_iter /= kshape_[i];
  }
  conv_in_channels_ = *im_channels_iter;
  std::swap(in_shape_, out_shape_);

  auto* x = X.template data<T, Context>();
  auto* y = Y->Reshape(Y_dims)->template mutable_data<T, Context>();
  for (int i = 0; i < X.dim(0); ++i) {
    this->Col2Im(x + i * X.stride(0), y + i * Y->stride(0));
  }
}

DEPLOY_CPU_OPERATOR(Im2Col);
REGISTER_CPU_OPERATOR(Im2ColGradient, Col2ImOp<CPUContext>);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Im2Col);
REGISTER_CUDA_OPERATOR(Im2ColGradient, Col2ImOp<CUDAContext>);
#endif

DEPLOY_CPU_OPERATOR(Col2Im);
REGISTER_CPU_OPERATOR(Col2ImGradient, Im2ColOp<CPUContext>);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Col2Im);
REGISTER_CUDA_OPERATOR(Col2ImGradient, Im2ColOp<CUDAContext>);
#endif

OPERATOR_SCHEMA(Im2Col).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Im2ColGradient).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Col2Im).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Col2ImGradient).NumInputs(1).NumOutputs(1);

REGISTER_GRADIENT(Im2Col, SimpleGradientMaker);
REGISTER_GRADIENT(Col2Im, SimpleGradientMaker);

} // namespace dragon
