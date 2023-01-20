#include "dragon/operators/array/one_hot_op.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void OneHotOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  vec64_t Y_dims(X.dims());
  Y_dims.push_back(depth_);

  // Set off values
  math::Set(
      X.count() * depth_,
      convert::To<T>(off_value_),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());

  // Set on values
  kernels::SetOneHot(
      X.count(),
      depth_,
      on_value_,
      X.template data<T, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(OneHot);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(OneHot);
#endif

OPERATOR_SCHEMA(OneHot).NumInputs(1).NumOutputs(1);

NO_GRADIENT(OneHot);

} // namespace dragon
