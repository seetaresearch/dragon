#include "dragon/operators/array/trilu_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {
template <class Context>
template <typename T>
void TriluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  kernels::SetTrilu(
      X.count(0, X.ndim() - 2),
      X.dim(-2),
      X.dim(-1),
      k_,
      upper_,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Trilu);
REGISTER_CPU_OPERATOR(TriluGradient, TriluOp<CPUContext>);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Trilu);
REGISTER_CUDA_OPERATOR(TriluGradient, TriluOp<CUDAContext>);
#endif

OPERATOR_SCHEMA(Trilu).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(TriluGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Trilu, SimpleGradientMaker);

} // namespace dragon
