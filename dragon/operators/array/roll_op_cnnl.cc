#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/roll_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLRollOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_shifts;
  shifts(0, &num_shifts);
  vec32_t X_shifts, X_axes;

  if (axes_.empty()) {
    X_shifts.push_back(shifts(0));
  } else {
    CHECK_EQ(num_shifts, int(axes_.size()))
        << "\nProviding " << axes_.size() << " dimensions and " << num_shifts
        << " shifts to roll.";
    for (int i = 0; i < axes_.size(); ++i) {
      int axis = axes_[i];
      axis = axis < 0 ? axis + X.ndim() : axis;
      CHECK(axis >= 0 && axis < X.ndim())
          << "\nExcepted the <axis> in [-" << X.ndim() << ", " << X.ndim()
          << "), got " << axes_[i] << ".";
      X_axes.push_back(axis);
      X_shifts.push_back(shifts(i));
    }
  }

  Output("X_shifts")->template CopyFrom<int>(X_shifts);
  CNNLSetTensorDesc<T>(input_desc_, X.dims());

  size_t scratch_size = 0;
  CNNL_CHECK(cnnlGetRollWorkspaceSize(
      ctx()->cnnl_handle(), input_desc_, &scratch_size));
  CNNL_CHECK(cnnlRoll(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      X_shifts.data(),
      X_shifts.size(),
      X_axes.data(),
      X_axes.size(),
      ctx()->workspace()->template data<Context>(scratch_size),
      scratch_size,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLRollGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);

  vec32_t Y_shifts, Y_axes;
  Input("X_shifts").template CopyTo<int>(Y_shifts);
  for (int i = 0; i < Y_shifts.size(); ++i) {
    Y_shifts[i] *= -1; // Reverse the shifts.
  }
  for (int i = 0; i < axes_.size(); ++i) {
    Y_axes.push_back(axes_[i] < 0 ? axes_[i] + dY.ndim() : axes_[i]);
  }

  CNNLSetTensorDesc<T>(input_desc_, dY.dims());

  size_t scratch_size = 0;
  CNNL_CHECK(cnnlGetRollWorkspaceSize(
      ctx()->cnnl_handle(), input_desc_, &scratch_size));
  CNNL_CHECK(cnnlRoll(
      ctx()->cnnl_handle(),
      input_desc_,
      dY.template data<T, Context>(),
      Y_shifts.data(),
      Y_shifts.size(),
      Y_axes.data(),
      Y_axes.size(),
      ctx()->workspace()->template data<Context>(scratch_size),
      scratch_size,
      input_desc_,
      dX->ReshapeLike(dY)->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(Roll);
DEPLOY_CNNL_OPERATOR(RollGradient);

DEFINE_OP_REPEATED_ARG(int64_t, CNNLRollOp, shifts);

} // namespace dragon

#endif // USE_MLU
