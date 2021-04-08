#include "dragon/operators/vision/space_to_depth_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SpaceToDepthOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  SET_INPUT_SPEC(0);

  int start_axis, end_axis, perm_count = 0;
  int num_dims = X.ndim(), num_axes = X.ndim() - 2;

  CHECK_GT(num_dims, 2) << "\nExcepted the spatial input"
                        << " with number of dimensions >= 3.";

  // Compute the reshape and transpose arguments
  vec64_t perm(size_t(num_axes * 2 + 2));
  vec64_t in_dims, in_shape = Input(0).dims();
  vec64_t out_shape = in_shape;

  if (data_format() == "NCHW") {
    start_axis = 2, end_axis = num_dims;
    out_shape[1] *= std::pow(block_size_, num_axes);
  } else if (data_format() == "NHWC") {
    start_axis = 1, end_axis = num_dims - 1;
    out_shape[end_axis] *= std::pow(block_size_, num_axes);
  } else {
    LOG(FATAL) << "Unknown DataFormat: " << data_format();
  }

  for (int i = 0; i < num_dims; i++) {
    if (i < start_axis) {
      in_dims.push_back(in_shape[i]);
      perm[i] = perm_count++;
    } else if (i >= start_axis && i < end_axis) {
      in_dims.push_back(in_shape[i] / block_size_);
      out_shape[i] = in_dims.back();
      in_dims.push_back(block_size_);
      perm[i] = perm_count++;
      perm[i + num_axes] = perm_count++;
    } else {
      in_dims.push_back(in_shape[i]);
      perm[perm_count] = perm_count;
      perm_count++;
    }
  }

  if (data_format() == "NCHW") {
    for (int i = 0; i < num_axes; i++) {
      perm.insert(perm.begin() + 1, perm.back());
      perm.pop_back(); // DCR mode
    }
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
  Buffer("X_strides")->template CopyFrom<int64_t>(x_strides);
  Buffer("Y_dims")->template CopyFrom<int64_t>(y_dims);

  kernels::Transpose(
      x_strides.size(),
      x_strides.data(),
      y_dims.data(),
      X.template data<T, Context>(),
      Y->Reshape(out_shape)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SpaceToDepthOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Generic>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(SpaceToDepth);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SpaceToDepth);
#endif

DEPLOY_CPU_OPERATOR(SpaceToDepthGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SpaceToDepthGradient);
#endif

OPERATOR_SCHEMA(SpaceToDepth)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SpaceToDepthGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(SpaceToDepth, SimpleGradientMaker);

} // namespace dragon
