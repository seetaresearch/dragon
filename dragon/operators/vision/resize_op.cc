#include "dragon/operators/vision/resize_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ResizeOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  Output("X_sizes")->template CopyFrom<int64_t>(in_dims_);
  Output("Y_sizes")->template CopyFrom<int64_t>(out_dims_);

  // Dispatch kernel according to mode and dimensions.
  if (mode_ == "NEAREST") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      kernels::ResizeNearest2d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          out_dims_.size() == 1 ? 1 : in_dims_[3],
          out_dims_[0],
          out_dims_.size() == 1 ? 1 : out_dims_[1],
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else if (out_dims_.size() == 3) {
      kernels::ResizeNearest3d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          in_dims_[4],
          out_dims_[0],
          out_dims_[1],
          out_dims_[2],
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "ResizeNearest" << out_dims_.size()
                 << "d is not supported.";
    }
  } else if (mode_ == "LINEAR") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      kernels::ResizeLinear2d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          out_dims_.size() == 1 ? 1 : in_dims_[3],
          out_dims_[0],
          out_dims_.size() == 1 ? 1 : out_dims_[1],
          align_corners_ > 0,
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "ResizeLinear" << out_dims_.size() << "d is not supported.";
    }
  } else {
    LOG(FATAL) << "Unknown interpolation mode: " << mode_;
  }
}

template <class Context>
template <typename T>
void ResizeGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));
  Input("X_sizes").template CopyTo<int64_t>(in_dims_);
  Input("Y_sizes").template CopyTo<int64_t>(out_dims_);

  auto* dy = dY.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();
  auto* dx_acc = (TypeMeta::Id<T>() == TypeMeta::Id<float>())
      ? (float*)nullptr
      : ctx()->workspace()->template data<float, Context>(dX->count());

  // Accumulate to dX.
  if (mode_ == "NEAREST") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      kernels::ResizeNearest2dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          out_dims_.size() == 1 ? 1 : in_dims_[3],
          out_dims_[0],
          out_dims_.size() == 1 ? 1 : out_dims_[1],
          data_format(),
          dy,
          dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
          ctx());
    } else if (out_dims_.size() == 3) {
      kernels::ResizeNearest3dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          in_dims_[4],
          out_dims_[0],
          out_dims_[1],
          out_dims_[2],
          data_format(),
          dy,
          dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
          ctx());
    } else {
      LOG(FATAL) << "ResizeNearest" << out_dims_.size()
                 << "d is not supported.";
    }
  } else if (mode_ == "LINEAR") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      kernels::ResizeLinear2dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          out_dims_.size() == 1 ? 1 : in_dims_[3],
          out_dims_[0],
          out_dims_.size() == 1 ? 1 : out_dims_[1],
          align_corners_ > 0,
          data_format(),
          dy,
          dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
          ctx());
    } else {
      LOG(FATAL) << "ResizeLinear" << out_dims_.size() << "d is not supported.";
    }
  } else {
    LOG(FATAL) << "Unknown interpolation mode: " << mode_;
  }

  // Convert to dX.
  if (dx_acc != nullptr) {
    math::Cast(dX->count(), dx_acc, dx, ctx());
  }
}

DEPLOY_CPU_OPERATOR(Resize);
DEPLOY_CPU_OPERATOR(ResizeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Resize);
DEPLOY_CUDA_OPERATOR(ResizeGradient);
#endif

OPERATOR_SCHEMA(Resize)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ResizeGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Resize, SimpleGradientMaker);

} // namespace dragon
