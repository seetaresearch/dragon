#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/slice_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLSliceOp<Context>::DoRunWithType() {
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

  vec64_t X_ends(X_starts);
  for (int i = 0; i < X_ends.size(); ++i) {
    X_ends[i] += Y_dims[i];
  }

  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNLSetTensorDesc<T>(output_desc_, Y_dims);
  CNNL_CHECK(cnnlStridedSlice(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      vec32_t({X_starts.begin(), X_starts.end()}).data(),
      vec32_t({X_ends.begin(), X_ends.end()}).data(),
      vec32_t(X_starts.size(), 1).data(),
      output_desc_,
      Y->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLSliceGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));

  vec64_t X_starts, Y_dims;
  Input("X_starts").template CopyTo<int64_t>(X_starts);
  Input("Y_dims").template CopyTo<int64_t>(Y_dims);

  // No slicing.
  if (dX->count() == dY.count()) {
    dX->CopyFrom(dY, ctx());
    return;
  }

  vec64_t X_ends(X_starts);
  for (int i = 0; i < X_ends.size(); ++i) {
    X_ends[i] += Y_dims[i];
  }

  CNNLSetTensorDesc<T>(input_desc_, Y_dims);
  CNNLSetTensorDesc<T>(output_desc_, dX->dims());
  CNNL_CHECK(cnnlStridedSliceBackward(
      ctx()->cnnl_handle(),
      vec32_t({X_starts.begin(), X_starts.end()}).data(),
      vec32_t({X_ends.begin(), X_ends.end()}).data(),
      vec32_t(X_starts.size(), 1).data(),
      input_desc_,
      dY.template data<T, Context>(),
      output_desc_,
      dX->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(Slice);
DEPLOY_CNNL_OPERATOR(SliceGradient);

DEFINE_OP_REPEATED_ARG(int64_t, CNNLSliceOp, starts);
DEFINE_OP_REPEATED_ARG(int64_t, CNNLSliceOp, sizes);

} // namespace dragon

#endif // USE_MLU
