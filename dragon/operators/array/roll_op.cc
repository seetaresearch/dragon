#include "dragon/operators/array/roll_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void RollOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto* X_ref = Buffer("X_ref")->ReshapeLike(X);
  if (axes_.empty()) X_ref->Reshape({X.count()});

  int num_shifts, num_dims = X_ref->ndim();
  vec64_t X_shifts(num_dims, 0);
  shifts(0, &num_shifts);

  if (axes_.empty()) {
    X_shifts[0] = shifts(0);
  } else {
    CHECK_EQ(num_shifts, int(axes_.size()))
        << "\nProviding " << axes_.size() << " dimensions and " << num_shifts
        << " shifts to roll.";
    for (int i = 0; i < axes_.size(); ++i) {
      int axis = axes_[i];
      axis = axis < 0 ? axis + num_dims : axis;
      CHECK(axis >= 0 && axis < num_dims)
          << "\nExcepted the <axis> in [-" << num_dims << ", " << num_dims
          << "), got " << axes_[i] << ".";
      X_shifts[axis] += shifts(i);
    }
  }
  Buffer("X_shifts")->template CopyFrom<int64_t>(X_shifts);

  kernels::Roll(
      num_dims,
      X_shifts.data(),
      X_ref->strides().data(),
      X_ref->dims().data(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void RollGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  auto* X_ref = Buffer("X_ref");

  vec64_t Y_shifts;
  Buffer("X_shifts")->template CopyTo<int64_t>(Y_shifts);
  for (int i = 0; i < Y_shifts.size(); ++i) {
    Y_shifts[i] *= -1; // Reverse the shifts.
  }

  kernels::Roll(
      X_ref->ndim(),
      Y_shifts.data(),
      X_ref->strides().data(),
      X_ref->dims().data(),
      dY.template data<T, Context>(),
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Roll);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Roll);
#endif

DEPLOY_CPU_OPERATOR(RollGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RollGradient);
#endif

OPERATOR_SCHEMA(Roll)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(RollGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Roll, SimpleGradientMaker);

} // namespace dragon
