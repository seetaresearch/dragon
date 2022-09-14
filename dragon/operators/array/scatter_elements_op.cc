#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/array/scatter_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ScatterElementsOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto &X_index = Input(1), &X_value = Input(2);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  CHECK_GT(X_index.count(), 0) << "\nLength of index must > 0.";
  CHECK_EQ(X.ndim(), X_index.ndim())
      << "\nMismatched number of dimensions between input and index.";
  CHECK_EQ(X_index.ndim(), X_value.ndim())
      << "\nMismatched number of dimensions between index and value.";
  for (int i = 0; i < X.ndim(); ++i) {
    CHECK_LE(X_index.dim(i), X_value.dim(i));
    if (i != axis) CHECK_LE(X_index.dim(i), X_value.dim(i));
  }

  // Copy the input data.
  Y->ReshapeLike(X)->CopyFrom(X, ctx());

  // Update with the new data.
  kernels::ScatterElements(
      axis,
      X.ndim(),
      X_index.dims().data(),
      X_value.strides().data(),
      X.strides().data(),
      X_index.template data<int64_t, Context>(),
      X_value.template data<T, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void ScatterElementsGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto *dX = Output(0), *dX_value = Output(1);
  GET_OP_AXIS_ARG(axis, dY.ndim(), 0);

  if (dX_value->has_name()) {
    kernels::GatherElements(
        axis,
        dY.ndim(),
        dY.strides().data(),
        X_index.dims().data(),
        X_index.template data<int64_t, Context>(),
        dY.template data<T, Context>(),
        dX_value->ReshapeLike(X_index)->template mutable_data<T, Context>(),
        ctx());
  }

  if (dX->has_name()) {
    dX->ReshapeLike(dY)->CopyFrom(dY, ctx());
    kernels::ScatterElements(
        axis,
        dY.ndim(),
        convert::To<T>(0.f),
        X_index.dims().data(),
        dY.strides().data(),
        X_index.template data<int64_t, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(ScatterElements);
DEPLOY_CPU_OPERATOR(ScatterElementsGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ScatterElements);
DEPLOY_CUDA_OPERATOR(ScatterElementsGradient);
#endif

OPERATOR_SCHEMA(ScatterElements)
    /* X, X_index, X_value */
    .NumInputs(3)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(ScatterElementsGradient)
    /* X_index, dY */
    .NumInputs(2)
    /* dX, dX_value */
    .NumOutputs(2)
    /* dY => dX */
    .AllowInplace({{1, 0}});

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(1), GO(0)}),
        vector<string>({GI(0), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(ScatterElements, GradientMaker);

} // namespace dragon
