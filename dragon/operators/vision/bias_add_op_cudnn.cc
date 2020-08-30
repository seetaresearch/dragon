#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/bias_add_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNBiasAddGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0), *dB = Output(1);

  int64_t N = dY.dim(0), C, S;
  if (data_format() == "NCHW") {
    C = dY.dim(1), S = dY.count(2);
  } else if (data_format() == "NHWC") {
    C = dY.dim(-1), S = dY.count(1) / dY.dim(-1);
  } else {
    LOG(FATAL) << "Unknown DataFormat: " << data_format();
  }

  if (dX->has_name()) {
    dX->ReshapeLike(dY)->CopyFrom(dY, ctx());
  }

  if (dB->has_name()) {
    vec64_t X_dims({N, S});
    X_dims.insert(X_dims.begin() + (data_format() == "NCHW" ? 1 : 2), C);
    CuDNNSetTensorDesc<T>(&input_desc_, X_dims, data_format());
    CuDNNSetBiasDesc<T>(&bias_desc_, 3, C, data_format());
    CUDNN_CHECK(cudnnConvolutionBackwardBias(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        input_desc_,
        dY.template data<T, Context>(),
        CuDNNType<T>::zero,
        bias_desc_,
        dB->Reshape({C})->template mutable_data<T, Context>()));
  }
}

template <class Context>
void CuDNNBiasAddGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CUDNN_OPERATOR(BiasAddGradient);

} // namespace dragon

#endif // USE_CUDNN
