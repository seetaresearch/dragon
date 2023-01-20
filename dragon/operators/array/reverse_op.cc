#include "dragon/operators/array/reverse_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ReverseOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_dims = X.ndim();
  vector<uint8_t> X_flips(num_dims, 0);
  for (int i = 0; i < axes_.size(); ++i) {
    int axis = axes_[i];
    axis = axis < 0 ? axis + num_dims : axis;
    CHECK(axis >= 0 && axis < num_dims)
        << "\nExcepted the <axis> in [-" << num_dims << ", " << num_dims
        << "), got " << axes_[i] << ".";
    X_flips[axis] = 1;
  }

  kernels::Reverse(
      num_dims,
      X_flips.data(),
      X.strides().data(),
      X.dims().data(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Reverse);
REGISTER_CPU_OPERATOR(ReverseGradient, ReverseOp<CPUContext>);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Reverse);
REGISTER_CUDA_OPERATOR(ReverseGradient, ReverseOp<CUDAContext>);
#endif

OPERATOR_SCHEMA(Reverse).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ReverseGradient).NumInputs(1).NumOutputs(1);

REGISTER_GRADIENT(Reverse, SimpleGradientMaker);

} // namespace dragon
