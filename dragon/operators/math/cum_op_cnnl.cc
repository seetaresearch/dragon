#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/math/cum_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLCumSumOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);
  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  size_t scratch_size;
  CNNL_CHECK(cnnlGetCumsumWorkspaceSize(
      ctx()->cnnl_handle(), input_desc_, axis, &scratch_size));
  CNNL_CHECK(cnnlCumsum_v2(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      axis,
      exclusive_,
      reverse_,
      CNNL_NOT_PROPAGATE_NAN,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<Context>(scratch_size),
      scratch_size));
}

template <class Context>
template <typename T>
void CNNLCumSumGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, dY.ndim(), 0);
  CNNLSetTensorDesc<T>(input_desc_, dY.dims());
  size_t scratch_size;
  CNNL_CHECK(cnnlGetCumsumWorkspaceSize(
      ctx()->cnnl_handle(), input_desc_, axis, &scratch_size));
  CNNL_CHECK(cnnlCumsum(
      ctx()->cnnl_handle(),
      input_desc_,
      dY.template data<T, Context>(),
      axis,
      exclusive_,
      !reverse_,
      CNNL_NOT_PROPAGATE_NAN,
      input_desc_,
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<Context>(scratch_size),
      scratch_size));
}

DEPLOY_CNNL_OPERATOR(CumSum);
DEPLOY_CNNL_OPERATOR(CumSumGradient);

} // namespace dragon

#endif // USE_MLU
