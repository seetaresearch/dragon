#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/space_to_depth_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLSpaceToDepthOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});

  int start_axis, end_axis, perm_count = 0;
  int num_dims = X.ndim(), num_axes = X.ndim() - 2;

  CHECK_GT(num_dims, 2) << "\nExcepted the spatial input"
                        << " with number of dimensions >= 3.";

  // Compute the reshape and transpose arguments.
  vec64_t perm(size_t(num_axes * 2 + 2));
  vec64_t in_dims, in_shape = X.dims();
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
      perm.insert(perm.begin() + (mode_ == "DCR" ? 1 : 2), perm.back());
      perm.pop_back();
    }
  }

  // Now, handle it as the generic transpose operation.
  Tensor X_reshape(in_dims);
  CHECK_EQ(X_reshape.count(), X.count())
      << "\nCould not rearrange " << X.DimString() << " to "
      << X_reshape.DimString() << " with block size " << block_size_ << ".";

  auto* buffer = ((void*)&X == (void*)Y)
      ? ctx()->workspace()->template data<T, Context>(X.count(), "BufferKernel")
      : Y->Reshape(out_shape)->template mutable_data<T, Context>();

  impl_.Setup<T>(X_reshape.dims(), perm, ctx());
  impl_.Compute<T>(
      X.template data<T, Context>(),
      buffer,
      ctx()->workspace()->template data<Context>(impl_.scratch_size()),
      ctx());

  if ((void*)&X == (void*)Y) {
    math::Copy(
        X.count(),
        buffer,
        Y->Reshape(out_shape)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void CNNLDepthToSpaceOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});

  int start_axis, end_axis;
  int num_dims = X.ndim(), num_axes = X.ndim() - 2;

  CHECK_GT(num_dims, 2) << "\nExcepted the spatial input"
                        << " with number of dimensions >= 3.";

  // Compute the reshape and transpose arguments.
  vec64_t perm(size_t(num_axes * 2 + 2), 0);
  vec64_t in_dims, out_shape = X.dims();

  if (data_format() == "NCHW") {
    start_axis = 2, end_axis = num_dims;
    out_shape[1] /= std::pow(block_size_, num_axes);
    in_dims = out_shape;
    perm[1] = (mode_ == "DCR" ? num_axes + 1 : 1);
    for (int i = 0; i < num_axes; i++) {
      perm[i * 2 + 2] = num_axes + i + 2;
      perm[i * 2 + 3] = i + (mode_ == "DCR" ? 1 : 2);
      in_dims.insert(in_dims.begin() + (mode_ == "DCR" ? 1 : 2), block_size_);
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

  // Now, handle it as the generic transpose operation.
  Tensor X_reshape(in_dims);
  CHECK_EQ(X_reshape.count(), X.count())
      << "\nCould not rearrange " << X.DimString() << " to "
      << X_reshape.DimString() << " with block size " << block_size_ << ".";

  auto* buffer = ((void*)&X == (void*)Y)
      ? ctx()->workspace()->template data<T, Context>(X.count(), "BufferKernel")
      : Y->Reshape(out_shape)->template mutable_data<T, Context>();

  impl_.Setup<T>(X_reshape.dims(), perm, ctx());
  impl_.Compute<T>(
      X.template data<T, Context>(),
      buffer,
      ctx()->workspace()->template data<Context>(impl_.scratch_size()),
      ctx());

  if ((void*)&X == (void*)Y) {
    math::Copy(
        X.count(),
        buffer,
        Y->Reshape(out_shape)->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CNNL_OPERATOR(SpaceToDepth);
DEPLOY_CNNL_OPERATOR(DepthToSpace);
REGISTER_CNNL_OPERATOR(SpaceToDepthGradient, CNNLDepthToSpaceOp<MLUContext>);
REGISTER_CNNL_OPERATOR(DepthToSpaceGradient, CNNLSpaceToDepthOp<MLUContext>);

} // namespace dragon

#endif // USE_MLU
