#include "dragon/core/workspace.h"
#include "dragon/operators/array/initialize_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void LinSpaceOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR((*Y));

  // Determine the generating range
  // Values are in a interval: [start, stop]
  int num_starts;
  start(0, &num_starts);
  vector<double> starts(num_starts), stops(num_starts);
  for (int i = 0; i < num_starts; ++i) {
    starts[i] = start(i);
    stops[i] = stop(i);
    CHECK_GT(stops[i], starts[i])
        << "\nInvalid generating range: "
        << "[" << starts[i] << ", " << stops[i] << "].";
  }

  kernel::LinSpace(
      Y->dim(0),
      Y->ndim() > 1 ? Y->dim(1) : 1,
      axis,
      starts.data(),
      stops.data(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void LinSpaceOp<Context>::RunOnDevice() {
  InitializeOp<Context>::RunOnDevice();
  DispatchHelper<NumericalTensorTypes>::Call(this);
}

DEPLOY_CPU_OPERATOR(LinSpace);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LinSpace);
#endif

OPERATOR_SCHEMA(LinSpace).NumInputs(0).NumOutputs(1);

NO_GRADIENT(LinSpace);

} // namespace dragon
