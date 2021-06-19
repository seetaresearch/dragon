#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/pool_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNPoolOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);

  // CuDNN NHWC pooling is slow.
  // Temporarily fallback to the naive implementation.
  if (data_format() == "NHWC" && mode_ == "AVG") {
    if (num_axes_ == 1 || num_axes_ == 2) {
      kernels::AvgPool2d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          num_axes_ == 1 ? 1 : in_dims_[3],
          out_dims_[2],
          num_axes_ == 1 ? 1 : out_dims_[3],
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1],
          pads_begin_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else if (num_axes_ == 3) {
      kernels::AvgPool3d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          in_dims_[4],
          out_dims_[2],
          out_dims_[3],
          out_dims_[4],
          kshape_[0],
          kshape_[1],
          kshape_[2],
          strides_[0],
          strides_[1],
          strides_[2],
          pads_begin_[0],
          pads_begin_[1],
          pads_begin_[2],
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "AvgPool" << num_axes_ << "d is not supported.";
    }
    return;
  }

  SetPoolDesc();
  CuDNNSetTensorDesc<T>(&input_desc_, X.dims(), data_format());
  CuDNNSetTensorDesc<T>(&output_desc_, out_shape_, data_format());

  CUDNN_CHECK(cudnnPoolingForward(
      ctx()->cudnn_handle(),
      pool_desc_,
      CuDNNType<T>::one,
      input_desc_,
      X.template data<T, Context>(),
      CuDNNType<T>::zero,
      output_desc_,
      Y->Reshape(out_shape_)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CuDNNPoolGradientOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *dX = Output(0);
  auto &Y = Input(1), &dY = Input(2);

  SetPoolDesc();
  CuDNNSetTensorDesc<T>(&input_desc_, dY.dims(), data_format());
  CuDNNSetTensorDesc<T>(&output_desc_, X.dims(), data_format());

  CUDNN_CHECK(cudnnPoolingBackward(
      ctx()->cudnn_handle(),
      pool_desc_,
      CuDNNType<T>::one,
      input_desc_,
      Y.template data<T, Context>(),
      input_desc_,
      dY.template data<T, Context>(),
      output_desc_,
      X.template data<T, Context>(),
      CuDNNType<T>::zero,
      output_desc_,
      dX->ReshapeLike(X)->template mutable_data<T, Context>()));
}

DEPLOY_CUDNN_OPERATOR(Pool);
DEPLOY_CUDNN_OPERATOR(PoolGradient);

} // namespace dragon

#endif // USE_CUDNN
