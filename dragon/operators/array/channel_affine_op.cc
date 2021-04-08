#include "dragon/operators/array/channel_affine_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ChannelAffineOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0, {0});
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);

  vec64_t affine_dims(
      {X.dims().begin() + axis, X.dims().begin() + end_axis + 1});
  CHECK(W.dims() == affine_dims)
      << "\nExcepted the weight shape is " << Tensor::DimString(affine_dims)
      << ", got " << W.DimString() << ".";
  if (InputSize() > 2) {
    CHECK(Input(2).dims() == affine_dims)
        << "\nExcepted the bias shape is " << Tensor::DimString(affine_dims)
        << ", got " << Input(2).DimString() << ".";
  }

  kernels::ChannelAffine(
      X.count(0, axis),
      X.count(end_axis + 1),
      X.count(axis, end_axis + 1),
      X.template data<T, Context>(),
      W.template data<T, Context>(),
      InputSize() <= 2 ? nullptr : Input(2).template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void ChannelAffineOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void ChannelAffineGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);

  vec32_t affine_dims = {(int)X.count(0, axis),
                         (int)X.count(axis, end_axis + 1),
                         (int)X.count(end_axis + 1)};
  vec32_t affine_axes = {0, 2};

  // dW = dY * X
  if (dW->has_name()) {
    Output(1)->ReshapeLike(Input(1));
    auto* x = Input(0).template data<T, Context>();
    auto* dw = Output(1)->template mutable_data<T, Context>();
    if (X.count() == W.count()) {
      math::Mul(
          X.count(),
          dY.template data<T, Context>(),
          X.template data<T, Context>(),
          dW->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx());
    } else {
      T* scratch =
          ctx()->workspace()->template data<T, Context>({X.count()})[0];
      math::Mul(
          X.count(),
          dY.template data<T, Context>(),
          X.template data<T, Context>(),
          scratch,
          ctx());
      math::ReduceSum(
          3,
          affine_dims.data(),
          2,
          affine_axes.data(),
          1.f,
          scratch,
          dW->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  // dB = dY
  if (dB->has_name()) {
    if (X.count() == W.count()) {
      dB->ReshapeLike(W)->CopyFrom(dY, ctx());
    } else {
      math::ReduceSum(
          3,
          affine_dims.data(),
          2,
          affine_axes.data(),
          1.f,
          dY.template data<T, Context>(),
          dB->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  // dX = dY * W
  if (dX->has_name()) {
    Output(0)->ReshapeLike(Input(-1));
    kernels::ChannelAffine(
        X.count(0, axis),
        X.count(end_axis + 1),
        X.count(axis, end_axis + 1),
        dY.template data<T, Context>(),
        W.template data<T, Context>(),
        (const T*)nullptr,
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void ChannelAffineGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(ChannelAffine);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ChannelAffine);
#endif

DEPLOY_CPU_OPERATOR(ChannelAffineGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ChannelAffineGradient);
#endif

OPERATOR_SCHEMA(ChannelAffine)
    /* X, W, B */
    .NumInputs(2, 3)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(ChannelAffineGradient)
    /* X, W, dY */
    .NumInputs(3)
    /* dX, dW, dB */
    .NumOutputs(3)
    /* dY => dX */
    .AllowInplace({{2, 0}});

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(ChannelAffine, GradientMaker);

} // namespace dragon
