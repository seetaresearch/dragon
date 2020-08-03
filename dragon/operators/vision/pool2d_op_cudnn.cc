#ifdef USE_CUDNN

#include "dragon/operators/vision/pool_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNPool2dOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);

  CuDNNSetTensorDesc<T>(&input_desc_, X.dims(), data_format());
  CuDNNSetTensorDesc<T>(&output_desc_, out_shape_, data_format());

  CUDNN_CHECK(cudnnSetPooling2dDescriptor(
      pool_desc_,
      pool_mode_,
      CUDNN_PROPAGATE_NAN,
      kshape_[0],
      kshape_[1],
      pad_l_[0],
      pad_l_[1],
      stride_[0],
      stride_[1]));

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
void CuDNNPool2dOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNPool2dGradientOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *dX = Output(0);
  auto &Y = Input(1), &dY = Input(2);

  CuDNNSetTensorDesc<T>(&input_desc_, dY.dims(), data_format());
  CuDNNSetTensorDesc<T>(&output_desc_, X.dims(), data_format());

  CUDNN_CHECK(cudnnSetPooling2dDescriptor(
      pool_desc_,
      pool_mode_,
      CUDNN_PROPAGATE_NAN,
      kshape_[0],
      kshape_[1],
      pad_l_[0],
      pad_l_[1],
      stride_[0],
      stride_[1]));

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

template <class Context>
void CuDNNPool2dGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CUDNN(Pool2d);
DEPLOY_CUDNN(Pool2dGradient);

} // namespace dragon

#endif // USE_CUDNN
