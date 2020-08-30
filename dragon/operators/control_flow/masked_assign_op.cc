#include "dragon/core/workspace.h"
#include "dragon/operators/control_flow/assign_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void MaskedAssignOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_mask = Input(1), *Y = Output(0);

  CHECK(X_mask.template IsType<bool>() || X_mask.template IsType<uint8_t>())
      << "\nExcepted bool or uint8 mask.";

  vec64_t X_dims, Y_dims;
  if (utils::math::IsBinaryBroadcast(X.dims(), X_mask.dims(), X_dims) &&
      utils::math::IsBinaryBroadcast(X_dims, Y->dims(), Y_dims) &&
      Y_dims == Y->dims()) {
    math::Where(
        X.ndim(),
        X.dims().data(),
        Y->ndim(),
        Y->dims().data(),
        X_mask.ndim(),
        X_mask.dims().data(),
        X.template data<T, Context>(),
        Y->template data<T, Context>(),
        (const bool*)X_mask.template raw_data<Context>(),
        Y->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Could not broadcast together with shapes: " << X.DimString()
               << " " << X_mask.DimString() << " " << Y->DimString();
  }
}

template <class Context>
void MaskedAssignOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(MaskedAssign);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(MaskedAssign);
#endif

OPERATOR_SCHEMA(MaskedAssign)
    /* X, M */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(MaskedAssign);

} // namespace dragon
