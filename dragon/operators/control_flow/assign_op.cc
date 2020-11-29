#include "dragon/core/workspace.h"
#include "dragon/operators/control_flow/assign_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void AssignOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_starts, num_sizes, num_dims = Y->ndim();
  vec64_t X_dims(num_dims), X_starts(num_dims);

  // Determine the interval of each dimension
  starts(0, &num_starts);
  sizes(0, &num_sizes);

  for (int i = 0; i < num_dims; i++) {
    auto dim_start = i < num_starts ? starts(i) : 0;
    auto dim_end = Y->dim(i);
    if (i < num_sizes) {
      auto dim_length = sizes(i);
      if (dim_length > 0) {
        dim_end = dim_start + dim_length;
      } else if (dim_length == 0) {
        dim_end = dim_start + 1;
      }
    }
    CHECK(dim_start >= 0 && dim_start < Y->dim(i))
        << "\nAssigning starts from " << dim_start << " of axis " << i << ", "
        << "while the dimension of this axis is " << Y->dim(i) << ".";
    CHECK(dim_end > 0 && dim_end <= Y->dim(i))
        << "\nAssigning ends at " << dim_end << " of axis " << i << ", "
        << "while the dimension of this axis is " << Y->dim(i) << ".";
    X_starts[i] = dim_start;
    X_dims[i] = dim_end - dim_start;
  }

  Tensor X_broadcast(X_dims);
  auto* x = X.template data<T, Context>();

  if (X.dims() != X_dims) {
    vec64_t dims1, dims2;
    if (math::utils::IsBinaryBroadcast(X.dims(), X_dims, dims1)) {
      CHECK(X_dims == dims1)
          << "\nCould not assign with shapes " << X.DimString() << " "
          << Tensor::DimString(X_dims);
      math::utils::ComputeBinaryBroadcastDims(X.dims(), X_dims, dims1, dims2);
      if (dims1 != dims2) {
        auto* scratch = ctx()->workspace()->template data<T, Context>(
            {X_broadcast.count()})[0];
        math::Set(
            X.ndim(),
            X.dims().data(),
            X_broadcast.ndim(),
            X_broadcast.dims().data(),
            x,
            scratch,
            ctx());
        x = scratch;
      }
    } else {
      LOG(FATAL) << "Could not broadcast together with shapes " << X.DimString()
                 << " " << Tensor::DimString(X_dims);
    }
  }

  kernel::Assign(
      num_dims,
      X_dims.data(),
      Y->strides().data(),
      X_starts.data(),
      x,
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void AssignOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Assign);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Assign);
#endif

OPERATOR_SCHEMA(Assign)
    /* V */
    .NumInputs(1)
    /* X */
    .NumOutputs(1);

NO_GRADIENT(Assign);

} // namespace dragon
