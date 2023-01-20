#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/sequence/rnn_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNRNNOp<Context>::DoRunWithType() {
  if (Input(0).dims() != input_dims_) {
    this->template SetOpDesc<T>();
  }

  if (InputSize() > 2) {
    INITIALIZE_TENSOR_VIA_SPEC(Input(2), hidden_dims_, T);
  }
  if (InputSize() > 3) {
    INITIALIZE_TENSOR_VIA_SPEC(Input(3), hidden_dims_, T);
  }

  Output(0)->Reshape(output_dims_);
  if (OutputSize() > 1) Output(1)->Reshape(hidden_dims_);
  if (OutputSize() > 2) Output(2)->Reshape(hidden_dims_);

  auto xAt = [this](int i) {
    if (i >= InputSize()) return (const T*)NULL;
    return Input(i).template data<T, Context>();
  };

  auto yAt = [this](int i) {
    if (i >= OutputSize()) return (T*)NULL;
    if (!Output(i)->has_name()) return (T*)NULL;
    return Output(i)->template mutable_data<T, Context>();
  };

  if (phase() == "TRAIN") {
    CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seqlen_,
        x_descs_->data(),
        &reserve_size_));
    auto* X_reserve = Output("X_reserve")->Reshape({int64_t(reserve_size_)});
    CUDNN_CHECK(cudnnRNNForwardTraining(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seqlen_,
        x_descs_->data(),
        xAt(0),
        hx_desc_,
        xAt(2),
        cx_desc_,
        xAt(3),
        weight_desc_,
        xAt(1),
        y_descs_->data(),
        yAt(0),
        hy_desc_,
        yAt(1),
        cy_desc_,
        yAt(2),
        ctx()->workspace()->template data<Context>(workspace_size_),
        workspace_size_,
        X_reserve->template mutable_data<uint8_t, Context>(),
        reserve_size_));
  } else if (phase() == "TEST") {
    CUDNN_CHECK(cudnnRNNForwardInference(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seqlen_,
        x_descs_->data(),
        xAt(0),
        hx_desc_,
        xAt(2),
        cx_desc_,
        xAt(3),
        weight_desc_,
        xAt(1),
        y_descs_->data(),
        yAt(0),
        hy_desc_,
        yAt(1),
        cy_desc_,
        yAt(2),
        ctx()->workspace()->template data<Context>(workspace_size_),
        workspace_size_));
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
template <typename T>
void CuDNNRNNGradientOp<Context>::DoRunWithType() {
  Output(0)->ReshapeLike(Input(0)); // dX
  Output(1)->ReshapeLike(Input(1)); // dW
  Output(2)->ReshapeLike(Input(2)); // dHx
  Output(3)->ReshapeLike(Input(3)); // dCx

  if (Input(0).dims() != input_dims_) {
    this->template SetOpDesc<T>();
  }

  auto xAt = [this](int i) {
    if (i >= InputSize()) return (const T*)NULL;
    if (!Input(i).has_name()) return (const T*)NULL;
    return Input(i).template data<T, Context>();
  };

  auto yAt = [this](int i) {
    if (i >= OutputSize()) return (T*)NULL;
    if (!Output(i)->has_name() && i > 0) return (T*)NULL;
    return Output(i)->template mutable_data<T, Context>();
  };

  CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
      ctx()->cudnn_handle(),
      rnn_desc_,
      seqlen_,
      x_descs_->data(),
      &reserve_size_));
  auto& X_reserve = Input("X_reserve");
  CHECK_EQ(reserve_size_, X_reserve.size());

  if (Output(0)->has_name() || Output(1)->has_name() || Output(2)->has_name() ||
      Output(3)->has_name()) {
    CUDNN_CHECK(cudnnRNNBackwardData(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seqlen_,
        y_descs_->data(),
        xAt(4), // Y
        y_descs_->data(),
        xAt(5), // dY
        hy_desc_,
        xAt(6), // dHy
        cy_desc_,
        xAt(7), // dCy
        weight_desc_,
        xAt(1), // W
        hx_desc_,
        xAt(2), // Hx
        cx_desc_,
        xAt(3), // Cx
        x_descs_->data(),
        yAt(0), // dX
        hx_desc_,
        yAt(2), // dHx
        cx_desc_,
        yAt(3), // dHy
        ctx()->workspace()->template data<Context>(workspace_size_),
        workspace_size_,
        X_reserve.template mutable_data<uint8_t, Context>(),
        reserve_size_));
  }

  if (Output(1)->has_name()) {
    // CuDNN accumulates the gradient of weights.
    // We should reset them before bakcward.
    math::Set(Output(1)->count(), convert::To<T>(0.f), yAt(1), ctx());
    CUDNN_CHECK(cudnnRNNBackwardWeights(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seqlen_,
        x_descs_->data(),
        xAt(0), // X
        hx_desc_,
        xAt(2), // Hx
        y_descs_->data(),
        xAt(4), // Y
        ctx()->workspace()->template data<Context>(workspace_size_),
        workspace_size_,
        weight_desc_,
        yAt(1), // dW
        X_reserve.template mutable_data<uint8_t, Context>(),
        reserve_size_));
  }
}

DEPLOY_CUDNN_OPERATOR(RNN);
DEPLOY_CUDNN_OPERATOR(RNNGradient);

} // namespace dragon

#endif // USE_CUDNN
