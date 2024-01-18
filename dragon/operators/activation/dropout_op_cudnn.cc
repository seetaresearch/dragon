#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/activation/dropout_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNDropoutOp<Context>::DoRunWithType() {
  if (TypeMeta::Id<T>() == TypeMeta::Id<bfloat16>()) {
    return DropoutOp<Context>::template DoRunWithType<T>();
  }
  auto &X = Input(0), *Y = Output(0, {0});
  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  } else if (phase() == "TRAIN") {
    const auto drop_ratio = this->ratio();
    if (drop_ratio != prev_ratio_) {
      CuDNNSetDropoutDesc(dropout_desc_, prev_ratio_ = drop_ratio, ctx());
    }
    CuDNNSetTensorDesc<T>(input_desc_, {X.count(), 1, 1, 1});
    size_t reserve_size;
    CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(input_desc_, &reserve_size));
    auto* X_reserve = Output("X_reserve")->Reshape({(int64_t)reserve_size});
    CUDNN_CHECK(cudnnDropoutForward(
        ctx()->cudnn_handle(),
        dropout_desc_,
        input_desc_,
        X.template data<T, Context>(),
        input_desc_,
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        X_reserve->template mutable_data<uint8_t, Context>(),
        reserve_size));
  } else {
    LOG(FATAL) << "Unsupported phase: " << phase();
  }
}

template <class Context>
template <typename T>
void CuDNNDropoutGradientOp<Context>::DoRunWithType() {
  if (TypeMeta::Id<T>() == TypeMeta::Id<bfloat16>()) {
    return DropoutGradientOp<Context>::template DoRunWithType<T>();
  }
  auto &dY = Input(0), *dX = Output(0);
  if (phase() == "TRAIN") {
    const auto drop_ratio = this->ratio();
    if (drop_ratio != prev_ratio_) {
      CuDNNSetDropoutDesc(dropout_desc_, prev_ratio_ = drop_ratio, ctx());
    }
    CuDNNSetTensorDesc<T>(input_desc_, {dY.count(), 1, 1, 1});
    CUDNN_CHECK(cudnnDropoutBackward(
        ctx()->cudnn_handle(),
        dropout_desc_,
        input_desc_,
        dY.template data<T, Context>(),
        input_desc_,
        dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
        Input("X_reserve").template mutable_data<uint8_t, Context>(),
        Input("X_reserve").size()));
  } else {
    LOG(FATAL) << "Unsupported phase: " << phase();
  }
}

DEPLOY_CUDNN_OPERATOR(Dropout);
DEPLOY_CUDNN_OPERATOR(DropoutGradient);

} // namespace dragon

#endif // USE_CUDNN
