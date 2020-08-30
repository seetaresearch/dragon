#include "dragon/core/workspace.h"
#include "dragon/operators/array/initialize_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void RangeOp<Context>::DoRunWithType() {
  // Determine the slice arguments
  int num_args;
  float start = 0.f, limit, delta;
  slice(0, &num_args);
  if (num_args == 2) {
    limit = slice(0), delta = slice(1);
  } else if (num_args == 3) {
    start = slice(0), limit = slice(1), delta = slice(2);
  } else {
    LOG(FATAL) << "Unexcepted number of slice arguments: " << num_args;
  }

  // Determine the generating range
  // Values are in a half-open interval: [start, stop)
  auto count = (int64_t)std::ceil((limit - start) / delta);
  CHECK_GT(count, 0) << "\nInvalid generating range: "
                     << "[" << start << ", " << limit
                     << ") with delta = " << delta << ".";

  kernel::Range(
      count,
      start,
      delta,
      Output(0)->Reshape({count})->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void RangeOp<Context>::RunOnDevice() {
  DispatchHelper<NumericalTensorTypes>::Call(this);
}

DEPLOY_CPU_OPERATOR(Range);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Range);
#endif

OPERATOR_SCHEMA(Range).NumInputs(0).NumOutputs(1);

NO_GRADIENT(Range);

} // namespace dragon
