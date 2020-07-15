#include "dragon/core/workspace.h"
#include "dragon/operators/array/initialize_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ArangeOp<Context>::DoRunWithType() {
  // Determine the slice arguments
  int num_args;
  float start = 0.f, stop, step;
  slice(0, &num_args);
  if (num_args == 2) {
    stop = slice(0), step = slice(1);
  } else if (num_args == 3) {
    start = slice(0), stop = slice(1), step = slice(2);
  } else {
    LOG(FATAL) << "Unexcepted number of slice arguments: " << num_args;
  }

  // Determine the generating range
  // Values are in a half-open interval: [start, stop)
  auto count = (int64_t)std::ceil((stop - start) / step);
  CHECK_GT(count, 0) << "\nInvalid generating range: "
                     << "[" << start << ", " << stop << ") with step = " << step
                     << ".";

  kernel::Arange(
      count,
      start,
      step,
      Output(0)->Reshape({count})->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void ArangeOp<Context>::RunOnDevice() {
  DispatchHelper<MathTensorTypes>::Call(this);
}

DEPLOY_CPU(Arange);
#ifdef USE_CUDA
DEPLOY_CUDA(Arange);
#endif

OPERATOR_SCHEMA(Arange).NumInputs(0).NumOutputs(1);

NO_GRADIENT(Arange);

} // namespace dragon
