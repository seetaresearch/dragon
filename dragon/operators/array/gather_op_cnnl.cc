#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/gather_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLGatherOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_index = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);

  const auto N = X.count(0, axis), S = X.count(end_axis + 1);
  const auto C = X.count(axis, end_axis + 1), K = X_index.count();
  CHECK_GT(K, 0) << "\nLength of index must > 0.";

  vec64_t X_dims(X.dims());
  vec64_t Y_dims(X_dims.begin(), X_dims.begin() + axis);
  Y_dims.insert(Y_dims.end(), X_index.dims().begin(), X_index.dims().end());
  Y_dims.insert(Y_dims.end(), X_dims.begin() + end_axis + 1, X_dims.end());

  CNNLSetTensorDesc<T>(input_desc_, {N, C, S});
  CNNLSetTensorDesc<int>(index_desc_, {K});
  CNNLSetTensorDesc<T>(output_desc_, {N, K, S});
  CNNL_CHECK(cnnlBatchGatherV2(
      ctx()->cnnl_handle(),
      1,
      0,
      input_desc_,
      X.template data<T, Context>(),
      index_desc_,
      X_index.template data<int, Context>(),
      output_desc_,
      Y->Reshape(Y_dims)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLGatherGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(Input("X_spec"));
  GET_OP_AXIS_ARG(axis, dX->ndim(), 0);
  GET_OP_AXIS_ARG(end_axis, dX->ndim(), axis);

  const auto N = dX->count(0, axis), S = dX->count(end_axis + 1);
  const auto C = dX->count(axis, end_axis + 1), K = X_index.count();

  auto* dy = dY.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();

  float *dy_acc = nullptr, *dx_acc = nullptr;
  if (TypeMeta::Id<T>() != TypeMeta::Id<float>()) {
    dy_acc = ctx()->workspace()->template data<float, Context>(
        dY.count() + dX->count());
    dx_acc = dy_acc + dY.count();
    math::Cast(dY.count(), dy, dy_acc, ctx());
  }

  // Zero dX.
  math::Set(
      dX->count(),
      0.f,
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      ctx());

  // Accumulate to dX.
  CNNLSetTensorDesc<float>(this->input_desc_, {N, K, S});
  CNNLSetTensorDesc<int>(this->index_desc_, {K});
  CNNLSetTensorDesc<float>(this->output_desc_, {N, C, S});
  CNNL_CHECK(cnnlIndexAdd(
      ctx()->cnnl_handle(),
      1,
      this->output_desc_,
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      this->index_desc_,
      X_index.template data<int, Context>(),
      this->input_desc_,
      dy_acc != nullptr ? dy_acc : reinterpret_cast<const float*>(dy),
      this->output_desc_,
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx)));

  // Convert to dX.
  if (dx_acc != nullptr) {
    math::Cast(dX->count(), dx_acc, dx, ctx());
  }
}

DEPLOY_CNNL_OPERATOR(Gather);
DEPLOY_CNNL_OPERATOR(GatherGradient);

} // namespace dragon

#endif // USE_MLU
