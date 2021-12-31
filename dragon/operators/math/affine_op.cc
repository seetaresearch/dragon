#include "dragon/operators/math/affine_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void AffineOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0, {0});

  // Compute affine dimensions.
  vec64_t affine_dims;
  for (auto axis : axes_) {
    axis = axis < 0 ? axis + X.ndim() : axis;
    CHECK(axis >= 0 && axis < X.ndim())
        << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim()
        << "), got " << axis << ".";
    affine_dims.push_back(X.dim(axis));
  }
  CHECK(W.dims() == affine_dims)
      << "\nExcepted the weight shape is " << Tensor::DimString(affine_dims)
      << ", got " << W.DimString() << ".";
  if (InputSize() > 2) {
    CHECK(Input(2).dims() == affine_dims)
        << "\nExcepted the bias shape is " << Tensor::DimString(affine_dims)
        << ", got " << Input(2).DimString() << ".";
  }

  math::Affine(
      X.ndim(),
      X.dims().data(),
      axes_.size(),
      axes_.data(),
      X.template data<T, Context>(),
      W.template data<T, Context>(),
      InputSize() <= 2 ? nullptr : Input(2).template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void AffineGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);

  // Compute reduce axes.
  vec64_t reduce_axes;
  for (int i = 0; i < X.ndim(); ++i) {
    bool keep = true;
    for (auto axis : axes_) {
      axis = axis < 0 ? axis + X.ndim() : axis;
      if (axis == i) keep = false;
    }
    if (keep) reduce_axes.push_back(i);
  }

  // Scratch to save the intermediates.
  T* data = nullptr;
  if (dW->has_name() && X.count() != W.count()) {
    data = ctx()->workspace()->template data<T, Context>(X.count());
  }

  // dW = dY * X
  if (dW->has_name()) {
    if (X.count() == W.count()) {
      math::Mul(
          X.count(),
          dY.template data<T, Context>(),
          X.template data<T, Context>(),
          dW->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Mul(
          X.count(),
          dY.template data<T, Context>(),
          X.template data<T, Context>(),
          data,
          ctx());
      math::ReduceSum(
          X.ndim(),
          X.dims().data(),
          reduce_axes.size(),
          reduce_axes.data(),
          1.f,
          data,
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
          X.ndim(),
          X.dims().data(),
          reduce_axes.size(),
          reduce_axes.data(),
          1.f,
          dY.template data<T, Context>(),
          dB->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  // dX = dY * W
  if (dX->has_name()) {
    math::Affine(
        X.ndim(),
        X.dims().data(),
        axes_.size(),
        axes_.data(),
        dY.template data<T, Context>(),
        W.template data<T, Context>(),
        (const T*)nullptr,
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(Affine);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Affine);
#endif

DEPLOY_CPU_OPERATOR(AffineGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(AffineGradient);
#endif

OPERATOR_SCHEMA(Affine)
    /* X, W, B */
    .NumInputs(2, 3)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(AffineGradient)
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

REGISTER_GRADIENT(Affine, GradientMaker);

} // namespace dragon
