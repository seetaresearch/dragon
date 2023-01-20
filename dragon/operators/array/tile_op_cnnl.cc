#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/tile_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLTileOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_repeats, num_dims, start_axis;
  repeats(0, &num_repeats);
  num_dims = std::max(num_repeats, X.ndim());
  start_axis = num_dims - X.ndim();

  vec64_t X_dims(num_dims, 1), Y_dims(num_dims, 1);
  std::copy(X.dims().begin(), X.dims().end(), X_dims.begin() + start_axis);
  std::copy(X.dims().begin(), X.dims().end(), Y_dims.begin() + start_axis);
  for (int i = 0; i < num_repeats; ++i) {
    start_axis = i + (num_dims - num_repeats);
    Y_dims[start_axis] = X_dims[start_axis] * repeats(i);
  }

  // Save for the gradient computation.
  Output("X_spec")->ReshapeLike(X);
  Output("X_dims")->template CopyFrom<int64_t>(X_dims);
  Output("Y_dims")->template CopyFrom<int64_t>(Y_dims);

  if (X_dims == Y_dims) {
    Y->Reshape(Y_dims)->CopyFrom(X, ctx());
    return;
  }

  CNNLSetTensorDesc<T>(input_desc_, X_dims);
  CNNLSetTensorDesc<T>(output_desc_, Y_dims);
  CNNL_CHECK(cnnlTile(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      output_desc_,
      Y->Reshape(Y_dims)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLTileGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));

  vec64_t X_dims, Y_dims;
  vec64_t Y_reduce_axes, Y_reduce_dims;
  Input("X_dims").template CopyTo<int64_t>(X_dims);
  Input("Y_dims").template CopyTo<int64_t>(Y_dims);
  for (int i = 0; i < X_dims.size(); ++i) {
    if (X_dims[i] != Y_dims[i]) {
      Y_reduce_axes.push_back(int64_t(Y_reduce_dims.size()));
      Y_reduce_dims.push_back(Y_dims[i] / X_dims[i]);
    }
    if (X_dims[i] != 1) {
      Y_reduce_dims.push_back(X_dims[i]);
    }
  }

  if (Y_reduce_axes.empty()) {
    dX->CopyFrom(dY, ctx());
    return;
  }

  reduce_impl_.Setup<T>(Y_reduce_dims, Y_reduce_axes, ctx());
  reduce_impl_.Compute<T>(
      dY.template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<Context>(reduce_impl_.scratch_size()),
      ctx());
}

DEPLOY_CNNL_OPERATOR(Tile);
DEPLOY_CNNL_OPERATOR(TileGradient);

DEFINE_OP_REPEATED_ARG(int64_t, CNNLTileOp, repeats);

} // namespace dragon

#endif // USE_MLU
