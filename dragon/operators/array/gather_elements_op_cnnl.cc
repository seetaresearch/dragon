#ifdef USE_MLU

#include "dragon/operators/array/gather_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLGatherElementsOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_index = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  CHECK_GT(X_index.count(), 0) << "\nLength of index must > 0.";
  CHECK_EQ(X.ndim(), X_index.ndim())
      << "\nMismatched number of dimensions between input and index.";
  for (int i = 0; i < X.ndim(); ++i) {
    if (i != axis) CHECK_EQ(X_index.dim(i), X.dim(i));
  }

  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNLSetTensorDesc<int>(index_desc_, X_index.dims());
  CNNLSetTensorDesc<T>(output_desc_, X_index.dims());

  CNNL_CHECK(cnnlGather(
      ctx()->cnnl_handle(),
      axis,
      input_desc_,
      X.template data<T, Context>(),
      index_desc_,
      X_index.template data<int, Context>(),
      output_desc_,
      Y->ReshapeLike(X_index)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLGatherElementsGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto &X_spec = Input("X_spec"), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, X_spec.ndim(), 0);

  auto* dx = dX->ReshapeLike(X_spec)->template mutable_data<T, Context>();
  math::Set(dX->count(), convert::To<T>(0.f), dx, ctx());

  CNNLSetTensorDesc<T>(this->input_desc_, dY.dims());
  CNNLSetTensorDesc<int>(this->index_desc_, X_index.dims());
  CNNLSetTensorDesc<T>(this->output_desc_, X_spec.dims());

  CNNL_CHECK(cnnlScatter(
      ctx()->cnnl_handle(),
      axis,
      this->output_desc_,
      dx,
      this->index_desc_,
      X_index.template data<int, Context>(),
      this->input_desc_,
      dY.template data<T, Context>(),
      this->output_desc_,
      dx,
      CNNL_SCATTER_ADD));
}

DEPLOY_CNNL_OPERATOR(GatherElements);
DEPLOY_CNNL_OPERATOR(GatherElementsGradient);

} // namespace dragon

#endif // USE_MLU
