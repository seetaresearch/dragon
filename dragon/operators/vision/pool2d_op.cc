#include "dragon/core/workspace.h"
#include "dragon/operators/vision/pool_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void Pool2dOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);

  if (mode_ == "MAX") {
    auto* Y_mask = Buffer("Y_mask")->Reshape(out_shape_);
    kernel::MaxPool2d(
        in_dims_[0],
        in_dims_[1],
        in_dims_[2],
        in_dims_[3],
        out_dims_[2],
        out_dims_[3],
        kshape_[0],
        kshape_[1],
        stride_[0],
        stride_[1],
        pad_l_[0],
        pad_l_[1],
        data_format(),
        X.template data<T, Context>(),
        Y_mask->template mutable_data<int, Context>(),
        Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
        ctx());
  } else if (mode_ == "AVG") {
    kernel::AvgPool2d(
        in_dims_[0],
        in_dims_[1],
        in_dims_[2],
        in_dims_[3],
        out_dims_[2],
        out_dims_[3],
        kshape_[0],
        kshape_[1],
        stride_[0],
        stride_[1],
        pad_l_[0],
        pad_l_[1],
        data_format(),
        X.template data<T, Context>(),
        Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void Pool2dOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float, double>>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void Pool2dGradientOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), &dY = Input(2), *dX = Output(0);

  if (mode_ == "MAX") {
    kernel::MaxPool2dGrad(
        in_dims_[0],
        in_dims_[1],
        in_dims_[2],
        in_dims_[3],
        out_dims_[2],
        out_dims_[3],
        kshape_[0],
        kshape_[1],
        stride_[0],
        stride_[1],
        pad_l_[0],
        pad_l_[1],
        data_format(),
        dY.template data<T, Context>(),
        Buffer("Y_mask")->template data<int, Context>(),
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else if (mode_ == "AVG") {
    kernel::AvgPool2dGrad(
        in_dims_[0],
        in_dims_[1],
        in_dims_[2],
        in_dims_[3],
        out_dims_[2],
        out_dims_[3],
        kshape_[0],
        kshape_[1],
        stride_[0],
        stride_[1],
        pad_l_[0],
        pad_l_[1],
        data_format(),
        dY.template data<T, Context>(),
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void Pool2dGradientOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float, double>>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Pool2d);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Pool2d);
#endif

DEPLOY_CPU_OPERATOR(Pool2dGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Pool2dGradient);
#endif

OPERATOR_SCHEMA(Pool2d)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(Pool2dGradient)
    /* X, Y, dY */
    .NumInputs(3)
    /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(0), O(0), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(Pool2d, GradientMaker);

} // namespace dragon
