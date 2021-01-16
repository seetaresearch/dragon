#include "dragon/operators/math/moments_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void MomentsOp<Context>::DoRunWithType() {
  using OutputT = typename math::AccmulatorType<T>::type;
  auto &X = Input(0), *Y1 = Output(0), *Y2 = Output(1);

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
    math::Cast(
        1,
        X.template data<T, Context>(),
        Y1->Reshape(Y_shape)->template mutable_data<OutputT, Context>(),
        ctx());
    math::Set(
        1,
        convert::To<OutputT>(0.f),
        Y2->Reshape(Y_shape)->template mutable_data<OutputT, Context>(),
        ctx());
  } else {
    kernel::Moments(
        X_dims.size(),
        X_dims.data(),
        reduce_axes.size(),
        reduce_axes.data(),
        X.template data<T, Context>(),
        Y1->Reshape(Y_shape)->template mutable_data<OutputT, Context>(),
        Y2->Reshape(Y_shape)->template mutable_data<OutputT, Context>(),
        ctx());
  }
}

template <class Context>
void MomentsOp<Context>::RunOnDevice() {
  DispatchHelper<NumericalTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Moments);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Moments);
#endif

OPERATOR_SCHEMA(Moments)
    /* X */
    .NumInputs(1)
    /* Mean, Var */
    .NumOutputs(2);

NO_GRADIENT(Moments);

} // namespace dragon
