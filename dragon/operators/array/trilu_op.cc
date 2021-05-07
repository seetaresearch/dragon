#include "dragon/operators/array/trilu_op.h"
#include "dragon/utils/op_kernels.h"

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

template <class Context>
void TriluOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Generic>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Trilu);
REGISTER_CPU_OPERATOR(TriluGradient, TriluOp<CPUContext>);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Trilu);
REGISTER_CUDA_OPERATOR(TriluGradient, TriluOp<CUDAContext>);
#endif

OPERATOR_SCHEMA(Trilu)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X -> Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(TriluGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY -> dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Trilu, SimpleGradientMaker);

} // namespace dragon
