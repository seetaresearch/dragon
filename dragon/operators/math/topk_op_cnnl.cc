#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/math/topk_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLTopKOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y_value = Output(0), *Y_index = Output(1);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  CHECK_LE(k_, C) << "\nThe top-K argument is out of the dimension.";
  auto Y_dims = X.dims();
  Y_dims[axis] = k_;

  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNLSetTensorDesc<T>(output_desc_, Y_dims);
  CNNLSetTensorDesc<int>(index_desc_, Y_dims);

  size_t scratch_size = 0;
  CNNL_CHECK(cnnlGetTopKTensorWorkspaceSize(
      ctx()->cnnl_handle(),
      input_desc_,
      k_,
      axis,
      largest_ > 0,
      output_desc_,
      index_desc_,
      &scratch_size));
  CNNL_CHECK(cnnlTopKTensor_v3(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      k_,
      axis,
      largest_ > 0,
      true, // sorted
      true, // lower index first
      ctx()->workspace()->template data<Context>(scratch_size),
      scratch_size,
      output_desc_,
      Y_value->Reshape(Y_dims)->template mutable_data<T, Context>(),
      index_desc_,
      Y_index->Reshape(Y_dims)->template mutable_data<int, Context>()));
}

DEPLOY_CNNL_OPERATOR(TopK);

} // namespace dragon

#endif // USE_MLU
