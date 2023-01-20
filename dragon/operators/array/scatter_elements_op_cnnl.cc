#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/scatter_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLScatterElementsOp<Context>::DoRunWithType() {
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

  CNNLSetTensorDesc<T>(input_desc_, X_index.dims());
  CNNLSetTensorDesc<int>(index_desc_, X_index.dims());
  CNNLSetTensorDesc<T>(output_desc_, X.dims());
  CNNL_CHECK(cnnlScatter(
      ctx()->cnnl_handle(),
      axis,
      output_desc_,
      Y->template data<T, Context>(),
      index_desc_,
      X_index.template data<int, Context>(),
      input_desc_,
      X_value.template data<T, Context>(),
      output_desc_,
      Y->template mutable_data<T, Context>(),
      CNNL_SCATTER));
}

template <class Context>
template <typename T>
void CNNLScatterElementsGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto *dX = Output(0), *dX_value = Output(1);
  GET_OP_AXIS_ARG(axis, dY.ndim(), 0);

  CNNLSetTensorDesc<T>(this->input_desc_, dY.dims());
  CNNLSetTensorDesc<int>(this->index_desc_, X_index.dims());
  CNNLSetTensorDesc<T>(this->output_desc_, X_index.dims());

  if (dX_value->has_name()) {
    CNNL_CHECK(cnnlGather(
        ctx()->cnnl_handle(),
        axis,
        this->input_desc_,
        dY.template data<T, Context>(),
        this->index_desc_,
        X_index.template data<int, Context>(),
        this->output_desc_,
        dX_value->ReshapeLike(X_index)->template mutable_data<T, Context>()));
  }

  if (dX->has_name()) {
    dX->ReshapeLike(dY)->CopyFrom(dY, ctx());
    auto* data = ctx()->workspace()->template data<T, Context>(X_index.count());
    math::Set(X_index.count(), convert::To<T>(0.f), data, ctx());
    CNNL_CHECK(cnnlScatter(
        ctx()->cnnl_handle(),
        axis,
        this->input_desc_,
        dX->template data<T, Context>(),
        this->index_desc_,
        X_index.template data<int, Context>(),
        this->output_desc_,
        data,
        this->input_desc_,
        dX->template mutable_data<T, Context>(),
        CNNL_SCATTER));
  }
}

DEPLOY_CNNL_OPERATOR(ScatterElements);
DEPLOY_CNNL_OPERATOR(ScatterElementsGradient);

} // namespace dragon

#endif // USE_MLU
