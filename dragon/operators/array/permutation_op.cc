#include "dragon/core/workspace.h"
#include "dragon/operators/array/initialize_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void PermutationOp<Context>::DoRunWithType() {
  auto* Y = Output(0)->Reshape({limit()});
  kernels::Permutation(
      Y->count(),
      Y->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<uint32_t, Context>(Y->count()),
      ctx());
}

template <class Context>
void PermutationOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Numerical>::Call(this);
}

DEPLOY_CPU_OPERATOR(Permutation);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Permutation);
#endif

OPERATOR_SCHEMA(Permutation).NumInputs(0).NumOutputs(1);

NO_GRADIENT(Permutation);

} // namespace dragon
