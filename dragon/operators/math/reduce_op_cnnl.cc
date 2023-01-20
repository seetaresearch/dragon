#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/math/reduce_op.h"

namespace dragon {

template <class Context, cnnlReduceOp_t Reducer>
template <typename T>
void CNNLReduceOp<Context, Reducer>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  // Compute reduce axes.
  vec64_t Y_dims(X.dims()), Y_shape(X.dims());
  vec64_t reduce_axes(axes_.begin(), axes_.end());
  if (axes_.empty()) {
    reduce_axes.resize(X.ndim());
    for (int i = 0; i < X.ndim(); ++i) {
      reduce_axes[i] = i;
    }
  }
  for (int i = 0; i < reduce_axes.size(); ++i) {
    auto axis = reduce_axes[i];
    reduce_axes[i] = axis = axis < 0 ? axis + X.ndim() : axis;
    CHECK(axis >= 0 && axis < X.ndim())
        << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim()
        << "), got " << axis << ".";
    Y_dims[axis] = 1;
    Y_shape[axis] = keep_dims_ ? 1 : -1;
  }

  // Squeeze output shape.
  const auto& erase_iter = std::remove_if(
      Y_shape.begin(), Y_shape.end(), [](int64_t x) { return x == -1; });
  Y_shape.erase(erase_iter, Y_shape.end());

  // Save context for gradient computation.
  Output("X_spec")->ReshapeLike(X);
  Output("Y_dims")->template CopyFrom<int64_t>(Y_dims);

  // Reduce or Copy X to Y.
  Y->Reshape(Y_shape);
  if (X.count() == 1 || X.count() == Y->count()) {
    Y->CopyFrom(X, ctx());
  } else {
    impl_.Setup<T>(X.dims(), reduce_axes, ctx());
    impl_.Compute<T>(
        X.template data<T, Context>(),
        Y->template mutable_data<T, Context>(),
        ctx()->workspace()->template data<Context>(impl_.scratch_size()),
        ctx());
  }
}

template <class Context>
template <typename T>
void CNNLReduceSumGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));
  if (dX->count() == 1 || dX->count() == dY.count()) {
    dX->CopyFrom(dY, ctx());
  } else {
    vec64_t Y_dims;
    Input("Y_dims").template CopyTo<int64_t>(Y_dims);
    CNNLSetTensorDesc<T>(input_desc_, Y_dims);
    CNNLSetTensorDesc<T>(output_desc_, dX->dims());
    CNNL_CHECK(cnnlExpand(
        ctx()->cnnl_handle(),
        input_desc_,
        dY.template data<T, Context>(),
        output_desc_,
        dX->template mutable_data<T, Context>()));
  }
}

template <class Context>
template <typename T>
void CNNLReduceMeanGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));
  if (dX->count() == 1 || dX->count() == dY.count()) {
    dX->CopyFrom(dY, ctx());
  } else {
    vec64_t Y_dims;
    Input("Y_dims").template CopyTo<int64_t>(Y_dims);
    CNNLSetTensorDesc<T>(input_desc_, Y_dims);
    CNNLSetTensorDesc<T>(output_desc_, dX->dims());
    const float scale = 1.f / float(dX->count() / dY.count());
    auto* data = ctx()->workspace()->template data<T, Context>({dY.count()});
    math::Scale(dY.count(), scale, dY.template data<T, Context>(), data, ctx());
    CNNL_CHECK(cnnlExpand(
        ctx()->cnnl_handle(),
        input_desc_,
        data,
        output_desc_,
        dX->template mutable_data<T, Context>()));
  }
}

DEPLOY_CNNL_OPERATOR(ReduceMax);
DEPLOY_CNNL_OPERATOR(ReduceMin);
DEPLOY_CNNL_OPERATOR(ReduceSum);
DEPLOY_CNNL_OPERATOR(ReduceMean);
DEPLOY_CNNL_OPERATOR(ReduceL1);
DEPLOY_CNNL_OPERATOR(ReduceL2);
DEPLOY_CNNL_OPERATOR(ReduceSumGradient);
DEPLOY_CNNL_OPERATOR(ReduceMeanGradient);

} // namespace dragon

#endif // USE_MLU
