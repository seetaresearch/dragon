#include "dragon/operators/array/assign_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void AssignOp<Context>::DoRunWithType() {
  auto &Y_ref = Input(0), &X = Input(1), *Y = Output(0, {0});

  // Determine the interval of each dimension
  int num_starts, num_sizes, num_dims = Y_ref.ndim();
  vec64_t X_dims(num_dims), X_starts(num_dims);
  starts(0, &num_starts);
  sizes(0, &num_sizes);

  for (int i = 0; i < num_dims; i++) {
    auto dim_start = i < num_starts ? starts(i) : 0;
    auto dim_end = Y_ref.dim(i);
    if (i < num_sizes) {
      auto dim_length = sizes(i);
      if (dim_length > 0) {
        dim_end = dim_start + dim_length;
      } else if (dim_length == 0) {
        dim_end = dim_start + 1;
      }
    }
    CHECK(dim_start >= 0 && dim_start < Y_ref.dim(i))
        << "\nAssigning starts from " << dim_start << " of axis " << i << ", "
        << "while the dimension of this axis is " << Y_ref.dim(i) << ".";
    CHECK(dim_end > 0 && dim_end <= Y_ref.dim(i))
        << "\nAssigning ends at " << dim_end << " of axis " << i << ", "
        << "while the dimension of this axis is " << Y_ref.dim(i) << ".";
    X_starts[i] = dim_start;
    X_dims[i] = dim_end - dim_start;
  }

  Tensor X_ref(X_dims);
  auto* data = X.template data<T, Context>();
  if (X.dims() != X_dims) {
    vec64_t dims1, dims2;
    if (math::utils::IsBinaryBroadcast(X.dims(), X_dims, dims1)) {
      CHECK(X_dims == dims1)
          << "\nCould not assign with shapes " << X.DimString() << " "
          << Tensor::DimString(X_dims);
      math::utils::ComputeBroadcastDims(X.dims(), X_dims, dims1, dims2);
      if (dims1 != dims2) {
        auto* new_data =
            ctx()->workspace()->template data<T, Context>(X_ref.count());
        math::Set(
            X.ndim(),
            X.dims().data(),
            X_ref.ndim(),
            X_ref.dims().data(),
            data,
            new_data,
            ctx());
        data = new_data;
      }
    } else {
      LOG(FATAL) << "Could not broadcast with shapes " << X.DimString() << " "
                 << Tensor::DimString(X_dims);
    }
  }

  // Copy the reference data.
  Y->ReshapeLike(Y_ref)->CopyFrom(Y_ref, ctx());

  // Update with the new data.
  kernels::Assign(
      num_dims,
      X_dims.data(),
      Y->strides().data(),
      X_starts.data(),
      data,
      Y->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Assign);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Assign);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Assign, Assign);
#endif

OPERATOR_SCHEMA(Assign)
    /* Y_ref, X */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1)
    /* Y_ref => Y */
    .AllowInplace({{0, 0}});

NO_GRADIENT(Assign);

} // namespace dragon
