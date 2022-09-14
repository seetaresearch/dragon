#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/array/scatter_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ScatterAddOp<Context>::DoRunWithType() {
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

  // Add the new data.
  kernels::ScatterAdd(
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
void ScatterAddOp<Context>::DoRunWithTypeAndCast() {
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

  // Copy input data.
  auto* scratch = ctx()->workspace()->template data<float, Context>(X.count());
  math::Cast(X.count(), X.template data<T, Context>(), scratch, ctx());

  // Add new data.
  kernels::ScatterAdd(
      axis,
      X.ndim(),
      X_index.dims().data(),
      X_value.strides().data(),
      X.strides().data(),
      X_index.template data<int64_t, Context>(),
      X_value.template data<T, Context>(),
      scratch,
      ctx());

  // Convert to Y.
  math::Cast(
      X.count(),
      scratch,
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void ScatterAddGradientOp<Context>::DoRunWithType() {
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
  }
}

DEPLOY_CPU_OPERATOR(ScatterAdd);
DEPLOY_CPU_OPERATOR(ScatterAddGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ScatterAdd);
DEPLOY_CUDA_OPERATOR(ScatterAddGradient);
#endif

OPERATOR_SCHEMA(ScatterAdd)
    /* X, X_index, X_value */
    .NumInputs(3)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(ScatterAddGradient)
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

REGISTER_GRADIENT(ScatterAdd, GradientMaker);

} // namespace dragon
