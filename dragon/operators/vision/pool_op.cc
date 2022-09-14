#include "dragon/operators/vision/pool_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void PoolOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);

  if (mode_ == "MAX") {
    auto* Y_mask = Output("Y_mask")->Reshape(out_shape_);
    if (num_axes_ == 1 || num_axes_ == 2) {
      kernels::MaxPool2d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          num_axes_ == 1 ? 1 : in_dims_[3],
          out_dims_[2],
          num_axes_ == 1 ? 1 : out_dims_[3],
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1],
          pads_begin_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          data_format(),
          X.template data<T, Context>(),
          Y_mask->template mutable_data<int, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else if (num_axes_ == 3) {
      kernels::MaxPool3d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          in_dims_[4],
          out_dims_[2],
          out_dims_[3],
          out_dims_[4],
          kshape_[0],
          kshape_[1],
          kshape_[2],
          strides_[0],
          strides_[1],
          strides_[2],
          pads_begin_[0],
          pads_begin_[1],
          pads_begin_[2],
          data_format(),
          X.template data<T, Context>(),
          Y_mask->template mutable_data<int, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "MaxPool" << num_axes_ << "d is not supported.";
    }
  } else if (mode_ == "AVG") {
    if (num_axes_ == 1 || num_axes_ == 2) {
      kernels::AvgPool2d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          num_axes_ == 1 ? 1 : in_dims_[3],
          out_dims_[2],
          num_axes_ == 1 ? 1 : out_dims_[3],
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1],
          pads_begin_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else if (num_axes_ == 3) {
      kernels::AvgPool3d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          in_dims_[4],
          out_dims_[2],
          out_dims_[3],
          out_dims_[4],
          kshape_[0],
          kshape_[1],
          kshape_[2],
          strides_[0],
          strides_[1],
          strides_[2],
          pads_begin_[0],
          pads_begin_[1],
          pads_begin_[2],
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "AvgPool" << num_axes_ << "d is not supported.";
    }
  }
}

template <class Context>
template <typename T>
void PoolGradientOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), &dY = Input(2), *dX = Output(0);

  if (mode_ == "MAX") {
    if (num_axes_ == 1 || num_axes_ == 2) {
      kernels::MaxPool2dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          num_axes_ == 1 ? 1 : in_dims_[3],
          out_dims_[2],
          num_axes_ == 1 ? 1 : out_dims_[3],
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1],
          pads_begin_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          data_format(),
          dY.template data<T, Context>(),
          const_cast<int*>(Input("Y_mask").template data<int, Context>()),
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    } else if (num_axes_ == 3) {
      kernels::MaxPool3dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          in_dims_[4],
          out_dims_[2],
          out_dims_[3],
          out_dims_[4],
          kshape_[0],
          kshape_[1],
          kshape_[2],
          strides_[0],
          strides_[1],
          strides_[2],
          pads_begin_[0],
          pads_begin_[1],
          pads_begin_[2],
          data_format(),
          dY.template data<T, Context>(),
          const_cast<int*>(Input("Y_mask").template data<int, Context>()),
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "MaxPool" << num_axes_ << "d is not supported.";
    }
  } else if (mode_ == "AVG") {
    if (num_axes_ == 1 || num_axes_ == 2) {
      kernels::AvgPool2dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          num_axes_ == 1 ? 1 : in_dims_[3],
          out_dims_[2],
          num_axes_ == 1 ? 1 : out_dims_[3],
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1],
          pads_begin_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          data_format(),
          dY.template data<T, Context>(),
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    } else if (num_axes_ == 3) {
      kernels::AvgPool3dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          in_dims_[4],
          out_dims_[2],
          out_dims_[3],
          out_dims_[4],
          kshape_[0],
          kshape_[1],
          kshape_[2],
          strides_[0],
          strides_[1],
          strides_[2],
          pads_begin_[0],
          pads_begin_[1],
          pads_begin_[2],
          data_format(),
          dY.template data<T, Context>(),
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "AvgPool" << num_axes_ << "d is not supported.";
    }
  }
}

DEPLOY_CPU_OPERATOR(Pool);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Pool);
#endif

DEPLOY_CPU_OPERATOR(PoolGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(PoolGradient);
#endif

OPERATOR_SCHEMA(Pool)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(PoolGradient)
    /* X, Y, dY */
    .NumInputs(3)
    /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), O(0), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(Pool, GradientMaker);

} // namespace dragon
