#include "dragon/operators/array/slice_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void SliceOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);

  int num_starts, num_sizes, num_dims = X.ndim();
  CHECK_GT(num_dims, 0) << "\nInvalid slice of a scalar.";
  vec64_t X_starts(num_dims), Y_dims(num_dims), Y_shape;

  // Compute the slice of each axis.
  vec32_t axes;
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
    if (Y_dims[i] != X.dim(i)) axes.push_back(i);
  }

  Output("X_starts")->template CopyFrom<int64_t>(X_starts);
  Output("Y_dims")->template CopyFrom<int64_t>(Y_dims);
  Y->Reshape(Y_shape);

  // No slicing.
  if (Y->count() == X.count()) {
    Y->CopyFrom(X, ctx());
    return;
  }

  // Slice along a single axis.
  if (axes.size() == 1) {
    const auto axis = axes[0];
    const auto copy_offset = X_starts[axis] * X.count(axis + 1);
    math::CopyMatrix(
        X.count(0, axis), // M
        Y_dims[axis] * X.count(axis + 1), // N
        X.count(axis), // ldx
        Y_dims[axis] * X.count(axis + 1), // ldy
        copy_offset, // x_offset
        0, // y_offset
        X.template data<T, Context>(),
        Y->template mutable_data<T, Context>(),
        ctx());
    return;
  }

  kernels::Slice(
      num_dims,
      X.strides().data(),
      Y_dims.data(),
      X_starts.data(),
      X.template data<T, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void SliceGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));

  vec64_t X_starts, Y_dims;
  Input("X_starts").template CopyTo<int64_t>(X_starts);
  Input("Y_dims").template CopyTo<int64_t>(Y_dims);

  vec32_t axes;
  for (int i = 0; i < dX->ndim(); ++i) {
    if (dX->dim(i) != Y_dims[i]) axes.push_back(i);
  }

  // No slicing.
  if (axes.empty()) {
    dX->CopyFrom(dY, ctx());
    return;
  }

  // Zero gradient of all positions.
  math::Set(
      dX->count(),
      convert::To<T>(0.f),
      dX->template mutable_data<T, Context>(),
      ctx());

  // Copy dY along a single slicing axis.
  if (axes.size() == 1) {
    const auto axis = axes[0];
    const auto copy_offset = X_starts[axis] * dX->count(axis + 1);
    math::CopyMatrix(
        dY.count(0, axis), // M
        dY.count(axis), // N
        dY.count(axis), // ldx
        dX->count(axis), // ldy
        0, // x_offset
        copy_offset, // y_offset
        dY.template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
    return;
  }

  // Copy dY along all slicing axes.
  kernels::SliceGrad(
      dX->ndim(),
      dX->strides().data(),
      Y_dims.data(),
      X_starts.data(),
      dY.template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Slice);
DEPLOY_CPU_OPERATOR(SliceGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Slice);
DEPLOY_CUDA_OPERATOR(SliceGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Slice, Slice);
DEPLOY_MPS_OPERATOR(SliceGradient, SliceGradient);
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
