#include "dragon/operators/array/reduce_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ReduceMaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  // Determine the reduce axes
  vec64_t Y_dims(X.dims()), Y_shape;
  vec32_t X_dims(Y_dims.begin(), Y_dims.end());
  vec32_t reduce_axes(axes_.begin(), axes_.end());
  if (axes_.empty()) {
    reduce_axes.resize(X.ndim());
    for (int i = 0; i < X.ndim(); ++i)
      reduce_axes[i] = i;
  }
  for (int i = 0; i < reduce_axes.size(); ++i) {
    auto axis = reduce_axes[i];
    reduce_axes[i] = axis = axis < 0 ? axis + X.ndim() : axis;
    CHECK(axis >= 0 && axis < X.ndim())
        << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim()
        << "), got " << axis << ".";
    Y_dims[axis] = 1;
  }

  // Squeeze the output shape if necessary
  for (int i = 0; i < X.ndim(); ++i) {
    if (keep_dims_ || Y_dims[i] != 1) Y_shape.push_back(Y_dims[i]);
  }

  if (X.count() == 1) {
    Y->Reshape(Y_shape)->CopyFrom(X, ctx());
  } else {
    math::ReduceMax(
        X_dims.size(),
        X_dims.data(),
        reduce_axes.size(),
        reduce_axes.data(),
        1.f,
        X.template data<T, Context>(),
        Y->Reshape(Y_shape)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void ReduceMaxOp<Context>::RunOnDevice() {
  DispatchHelper<NumericalTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(ReduceMax);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ReduceMax);
#endif

OPERATOR_SCHEMA(ReduceMax)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(ReduceMax);

} // namespace dragon
