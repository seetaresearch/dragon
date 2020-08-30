#include "dragon/operators/control_flow/copy_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CopyOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  math::Copy(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void CopyOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Copy);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Copy);
#endif

OPERATOR_SCHEMA(Copy)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(Copy);

} // namespace dragon
