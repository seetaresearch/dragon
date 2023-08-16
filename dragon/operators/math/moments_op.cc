#include "dragon/operators/math/moments_op.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MomentsOp<Context>::DoRunWithType() {
  using AccT = typename math::Traits<T>::accumulator_type;
  auto &X = Input(0), *Y_mean = Output(0), *Y_var = Output(1);

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

  if (X.count() == 1) {
    math::Cast(
        1,
        X.template data<T, Context>(),
        Y_mean->Reshape(Y_shape)->template mutable_data<AccT, Context>(),
        ctx());
    math::Set(
        1,
        convert::To<AccT>(0.f),
        Y_var->Reshape(Y_shape)->template mutable_data<AccT, Context>(),
        ctx());
  } else {
    kernels::Moments(
        X.ndim(),
        X.dims().data(),
        reduce_axes.size(),
        reduce_axes.data(),
        X.template data<T, Context>(),
        Y_mean->Reshape(Y_shape)->template mutable_data<AccT, Context>(),
        Y_var->Reshape(Y_shape)->template mutable_data<AccT, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(Moments);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Moments);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Moments, Moments);
#endif

OPERATOR_SCHEMA(Moments).NumInputs(1).NumOutputs(2);

NO_GRADIENT(Moments);

} // namespace dragon
