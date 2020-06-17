#include "dragon/operators/vision/depth_to_space_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void DepthToSpaceOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int start_axis, end_axis;
  int num_dims = X.ndim(), num_axes = X.ndim() - 2;

  CHECK_GT(num_dims, 2) << "\nExcepted the spatial input"
                        << " with number of dimensions >= 3.";

  // Compute the reshape and transpose arguments
  vec64_t perm(size_t(num_axes * 2 + 2), 0);
  vec64_t in_dims, out_shape = X.dims();

  if (data_format() == "NCHW") {
    start_axis = 2, end_axis = num_dims;
    out_shape[1] /= std::pow(block_size_, num_axes);
    in_dims = out_shape;
    perm[1] = num_axes + 1;
    for (int i = 0; i < num_axes; i++) {
      perm[i * 2 + 2] = num_axes + i + 2;
      perm[i * 2 + 3] = i + 1;
      in_dims.insert(in_dims.begin() + 1, block_size_);
      out_shape[start_axis + i] *= block_size_;
    }
  } else if (data_format() == "NHWC") {
    start_axis = 1, end_axis = num_dims - 1;
    out_shape[end_axis] /= std::pow(block_size_, num_axes);
    in_dims = out_shape;
    for (int i = 0; i < num_axes; i++) {
      perm[i * 2 + 1] = i + 1;
      perm[i * 2 + 2] = num_axes + i + 1;
      in_dims.insert(in_dims.begin() + num_axes + 1, block_size_);
      out_shape[start_axis + i] *= block_size_;
    }
    perm.back() = perm.size() - 1;
  } else {
    LOG(FATAL) << "Unknown DataFormat: " << data_format();
  }

  // Now, handle it as the generic transpose operation
  Tensor X_reshape(in_dims);
  vec64_t x_strides(in_dims.size()), y_dims(in_dims.size());

  CHECK_EQ(X_reshape.count(), X.count())
      << "\nCould not rearrange " << X.DimString() << " to "
      << X_reshape.DimString() << " with block size " << block_size_ << ".";

  for (int i = 0; i < in_dims.size(); i++) {
    x_strides[i] = X_reshape.stride(perm[i]);
    y_dims[i] = X_reshape.dim(perm[i]);
  }

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);
  Buffer("X_strides")->template CopyFrom<int64_t>(x_strides);
  Buffer("Y_dims")->template CopyFrom<int64_t>(y_dims);

  kernel::Transpose(
      x_strides.size(),
      x_strides.data(),
      y_dims.data(),
      X.template data<T, Context>(),
      Y->Reshape(out_shape)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void DepthToSpaceOp<Context>::RunOnDevice() {
  DispatchHelper<AllTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(DepthToSpace);
#ifdef USE_CUDA
DEPLOY_CUDA(DepthToSpace);
#endif

DEPLOY_CPU(DepthToSpaceGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(DepthToSpaceGradient);
#endif

OPERATOR_SCHEMA(DepthToSpace)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(DepthToSpaceGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(DepthToSpace, SimpleGradientMaker);

} // namespace dragon
