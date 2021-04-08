#include "dragon/operators/array/triangular_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {
template <class Context>
template <typename T>
void TriangularOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  kernels::SetTriangular(
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
void TriangularOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Generic>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Triangular);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Triangular);
#endif

OPERATOR_SCHEMA(Triangular)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X -> Y */
    .AllowInplace({{0, 0}});

NO_GRADIENT(Triangular);

} // namespace dragon
