#include "dragon/operators/array/slice_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SliceOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_starts, num_sizes, num_dims = X.ndim();
  CHECK_GT(num_dims, 0) << "\nInvalid slice of a scalar.";
  vec64_t X_starts(num_dims), Y_dims(num_dims), Y_shape;

  // Determine the interval of each dimension
  starts(0, &num_starts);
  sizes(0, &num_sizes);

  for (int i = 0; i < num_dims; i++) {
    auto dim_start = i < num_starts ? starts(i) : 0;
    auto dim_end = X.dim(i);
    auto keep_dim = true;
    if (i < num_sizes) {
      auto dim_length = sizes(i);
      if (dim_length > 0) {
        dim_end = dim_start + dim_length;
      } else if (dim_length == 0) {
        keep_dim = false;
        dim_end = dim_start + 1;
      }
    }
    CHECK(dim_start >= 0 && dim_start < X.dim(i))
        << "\nSlicing starts from " << dim_start << " of axis " << i << ", "
        << "while the dimension of this axis is " << X.dim(i) << ".";
    CHECK(dim_end > 0 && dim_end <= X.dim(i))
        << "\nSlicing ends at " << dim_end << " of axis " << i << ", "
        << "while the dimension of this axis is " << X.dim(i) << ".";
    X_starts[i] = dim_start;
    Y_dims[i] = dim_end - dim_start;
    if (keep_dim) Y_shape.push_back(Y_dims[i]);
  }

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);
  Buffer("X_starts")->template CopyFrom<int64_t>(X_starts);
  Buffer("Y_dims")->template CopyFrom<int64_t>(Y_dims);

  // Maybe just copy the contents
  Y->Reshape(Y_shape);
  if (Y->count() == X.count()) {
    Y->CopyFrom(X, ctx());
    return;
  }

  kernel::Slice(
      num_dims,
      X.strides().data(),
      Y_dims.data(),
      X_starts.data(),
      X.template data<T, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SliceOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SliceGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);

  // Maybe just copy the contents
  dX->ReshapeLike(RESTORE_INPUT_SPEC(0));
  if (dX->count() == dY.count()) {
    dX->CopyFrom(dY, ctx());
    return;
  }

  vec64_t X_starts, Y_dims;
  Buffer("X_starts")->template CopyTo<int64_t>(X_starts);
  Buffer("Y_dims")->template CopyTo<int64_t>(Y_dims);

  // Zero the redundant gradients
  auto* dx = dX->template mutable_data<T, Context>();
  math::Set(dX->count(), cast::to<T>(0.f), dx, ctx());

  // Copy the dY to the right positions
  kernel::SliceGrad(
      dX->ndim(),
      dX->strides().data(),
      Y_dims.data(),
      X_starts.data(),
      dY.template data<T, Context>(),
      dx,
      ctx());
}

template <class Context>
void SliceGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Slice);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Slice);
#endif

DEPLOY_CPU_OPERATOR(SliceGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SliceGradient);
#endif

OPERATOR_SCHEMA(Slice)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SliceGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Slice, SimpleGradientMaker);

} // namespace dragon
