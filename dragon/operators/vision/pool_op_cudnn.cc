#ifdef USE_CUDNN

#include "dragon/operators/vision/pool_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNPoolOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);

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
void CuDNNPoolOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
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

template <class Context>
void CuDNNPoolGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CUDNN_OPERATOR(Pool);
DEPLOY_CUDNN_OPERATOR(PoolGradient);

} // namespace dragon

#endif // USE_CUDNN
