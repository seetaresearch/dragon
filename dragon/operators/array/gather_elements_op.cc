#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/array/gather_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void GatherElementsOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_index = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  CHECK_GT(X_index.count(), 0) << "\nLength of index must > 0.";
  CHECK_EQ(X.ndim(), X_index.ndim())
      << "\nMismatched number of dimensions between input and index.";
  for (int i = 0; i < X.ndim(); ++i) {
    if (i != axis) CHECK_EQ(X_index.dim(i), X.dim(i));
  }

  kernels::GatherElements(
      axis,
      X.ndim(),
      X.strides().data(),
      X_index.dims().data(),
      X_index.template data<int64_t, Context>(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X_index)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void GatherElementsGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto &X_spec = Input("X_spec"), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, X_spec.ndim(), 0);

  auto* dx = dX->ReshapeLike(X_spec)->template mutable_data<T, Context>();
  auto* dx_acc = (TypeMeta::Id<T>() == TypeMeta::Id<float>())
      ? (float*)nullptr
      : ctx()->workspace()->template data<float, Context>(dX->count());

  // Empty gradient.
  math::Set(
      dX->count(),
      0.f,
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      ctx());

  // Scatter and accumulate to dX.
  kernels::ScatterAdd(
      axis,
      X_spec.ndim(),
      X_index.dims().data(),
      X_index.strides().data(),
      X_spec.strides().data(),
      X_index.template data<int64_t, Context>(),
      dY.template data<T, Context>(),
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      ctx());

  // Convert to dX.
  if (dx_acc != nullptr) {
    math::Cast(dX->count(), dx_acc, dx, ctx());
  }
}

DEPLOY_CPU_OPERATOR(GatherElements);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GatherElements);
#endif

DEPLOY_CPU_OPERATOR(GatherElementsGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GatherElementsGradient);
#endif

OPERATOR_SCHEMA(GatherElements)
    /* X, X_index */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(GatherElementsGradient)
    /* X_index, dY */
    .NumInputs(2)
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
        vector<string>({I(1), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(GatherElements, GradientMaker);

} // namespace dragon
