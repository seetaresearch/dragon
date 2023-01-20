#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/expand_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLExpandOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);

  int num_dims;
  dims(0, &num_dims);

  vec64_t X_dims(num_dims), Y_dims;
  for (int i = 0; i < num_dims; ++i) {
    const auto new_dim = dims(i);
    X_dims[i] = (new_dim < 0 ? X.dim(i - num_dims) : new_dim);
  }

  CHECK(math::utils::IsBinaryBroadcast(X.dims(), X_dims, Y_dims))
      << "\nCould not broadcast together with shapes: " << X.DimString() << " "
      << Tensor::DimString(X_dims);

  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNLSetTensorDesc<T>(output_desc_, Y_dims);
  CNNL_CHECK(cnnlExpand(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      output_desc_,
      Y->Reshape(Y_dims)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLExpandGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));

  vec64_t broadcast_axes, _;
  math::utils::ComputeBroadcastAxes(
      dX->dims(), dY.dims(), dY.dims(), broadcast_axes, _);

  if (broadcast_axes.empty()) {
    dX->CopyFrom(dY, ctx());
  } else {
    reduce_impl_.Setup<T>(dY.dims(), broadcast_axes, ctx());
    reduce_impl_.Compute<T>(
        dY.template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx()->workspace()->template data<Context>(reduce_impl_.scratch_size()),
        ctx());
  }
}

DEPLOY_CNNL_OPERATOR(Expand);
DEPLOY_CNNL_OPERATOR(ExpandGradient);

DEFINE_OP_REPEATED_ARG(int64_t, CNNLExpandOp, dims);

} // namespace dragon

#endif // USE_MLU
