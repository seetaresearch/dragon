#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/assign_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLAssignOp<Context>::DoRunWithType() {
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
    CHECK(math::utils::IsBinaryBroadcast(X.dims(), X_dims, dims1))
        << "\nCould not broadcast with shapes " << X.DimString() << " "
        << Tensor::DimString(X_dims);
    CHECK(X_dims == dims1) << "\nCould not assign with shapes " << X.DimString()
                           << " " << Tensor::DimString(X_dims);
    math::utils::ComputeBroadcastDims(X.dims(), X_dims, dims1, dims2);
    if (dims1 != dims2) {
      T* scratch = ctx()->workspace()->template data<T, Context>(X_ref.count());
      math::Set(
          X.ndim(),
          X.dims().data(),
          X_ref.ndim(),
          X_ref.dims().data(),
          data,
          scratch,
          ctx());
      data = scratch;
    }
  }

  // Copy the reference data.
  Y->ReshapeLike(Y_ref)->CopyFrom(Y_ref, ctx());

  vec64_t X_ends(X_starts);
  for (int i = 0; i < X_ends.size(); ++i) {
    X_ends[i] += X_dims[i];
  }

  auto* y_slice = ctx()->workspace()->template data<T, Context>(
      X_ref.count() + Y_ref.count(), "BufferKernel");
  auto* y_buffer = y_slice + X_ref.count();

  CNNLSetTensorDesc<T>(input_desc_, X_dims);
  CNNLSetTensorDesc<T>(output_desc_, Y->dims());
  CNNL_CHECK(cnnlStridedSlice(
      ctx()->cnnl_handle(),
      output_desc_,
      Y_ref.template data<T, Context>(),
      vec32_t({X_starts.begin(), X_starts.end()}).data(),
      vec32_t({X_ends.begin(), X_ends.end()}).data(),
      vec32_t(X_starts.size(), 1).data(),
      input_desc_,
      y_slice));
  math::Sub(X_ref.count(), data, y_slice, y_slice, ctx());
  CNNL_CHECK(cnnlStridedSliceBackward(
      ctx()->cnnl_handle(),
      vec32_t({X_starts.begin(), X_starts.end()}).data(),
      vec32_t({X_ends.begin(), X_ends.end()}).data(),
      vec32_t(X_starts.size(), 1).data(),
      input_desc_,
      y_slice,
      output_desc_,
      y_buffer));
  math::Add(
      Y_ref.count(),
      y_buffer,
      Y->template data<T, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CNNL_OPERATOR(Assign);
DEFINE_OP_REPEATED_ARG(int64_t, CNNLAssignOp, starts);
DEFINE_OP_REPEATED_ARG(int64_t, CNNLAssignOp, sizes);

} // namespace dragon

#endif // USE_MLU
