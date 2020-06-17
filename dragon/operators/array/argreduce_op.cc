#include "dragon/operators/array/argreduce_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ArgReduceOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  // Determine the reduce scheme
  // 1) Reduce along the specified axis
  // 2) Reduce to a scalar
  int64_t outer_dim, axis_dim, inner_dim;
  if (axis != INT_MAX) {
    axis_dim = X.dim(axis);
    outer_dim = X.count(0, axis);
    inner_dim = X.count(axis + 1);
  } else {
    axis_dim = X.count();
    outer_dim = inner_dim = 1;
  }

  // Determine the output dimensions
  auto Y_dims = X.dims();
  if (!keep_dims_) {
    if (axis != INT_MAX) {
      if (top_k_ > 1) {
        Y_dims[axis] = top_k_;
      } else {
        Y_dims.erase(Y_dims.begin() + axis);
      }
    } else {
      if (top_k_ > 1) {
        Y_dims = {top_k_};
      } else {
        Y_dims = {};
      }
    }
  } else {
    if (axis != INT_MAX) {
      Y_dims[axis] = top_k_;
    } else {
      Y_dims = {top_k_};
    }
  }

  // <DeviceSort> is required dispatch the top-k argreudce.
  // We will implement the generic kernels in the future.
  if (top_k_ != 1) {
    CPUContext context;
    if (operation_ == "MAX") {
      kernel::ArgMax(
          outer_dim,
          inner_dim,
          axis_dim,
          top_k_,
          X.template data<T, CPUContext>(),
          Y->Reshape(Y_dims)->template mutable_data<int64_t, CPUContext>(),
          &context);
    } else if (operation_ == "MIN") {
      kernel::ArgMin(
          outer_dim,
          inner_dim,
          axis_dim,
          top_k_,
          X.template data<T, CPUContext>(),
          Y->Reshape(Y_dims)->template mutable_data<int64_t, CPUContext>(),
          &context);
    } else {
      LOG(FATAL) << "Unknown operation: " << operation_;
    }
  } else {
    if (operation_ == "MAX") {
      kernel::ArgMax(
          outer_dim,
          inner_dim,
          axis_dim,
          1,
          X.template data<T, Context>(),
          Y->Reshape(Y_dims)->template mutable_data<int64_t, Context>(),
          ctx());
    } else if (operation_ == "MIN") {
      kernel::ArgMin(
          outer_dim,
          inner_dim,
          axis_dim,
          1,
          X.template data<T, Context>(),
          Y->Reshape(Y_dims)->template mutable_data<int64_t, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "Unknown operation: " << operation_;
    }
  }
}

template <class Context>
void ArgReduceOp<Context>::RunOnDevice() {
  DispatchHelper<AllTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(ArgReduce);
#ifdef USE_CUDA
DEPLOY_CUDA(ArgReduce);
#endif

OPERATOR_SCHEMA(ArgReduce)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(ArgReduce);

} // namespace dragon
