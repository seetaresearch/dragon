#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/activation/dropout_op.h"

#if CUDNN_VERSION_MIN(7, 0, 0)

namespace dragon {

template <class Context>
template <typename T>
void CuDNNDropoutOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  CuDNNSetTensorDesc<T>(&input_desc_, {X.count(), 1, 1, 1});
  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  } else if (phase() == "TRAIN") {
    // Initialize the dropout states
    if (!states_initialized_) {
      states_initialized_ = true;
      size_t states_size;
      CUDNN_CHECK(
          cudnnDropoutGetStatesSize(ctx()->cudnn_handle(), &states_size));
      std::lock_guard<std::mutex> lk(CUDAContext::mutex());
      auto* X_states = workspace()->CreateTensor(
          "/share/cudnn/dropout:" + str::to(rng_seed_) + "/states");
      if (X_states->count() > 0) {
        CUDNN_CHECK(cudnnRestoreDropoutDescriptor(
            dropout_desc_,
            ctx()->cudnn_handle(),
            this->ratio(),
            X_states->template mutable_data<uint8_t, Context>(),
            states_size,
            rng_seed_));
      } else {
        X_states->Reshape({(int64_t)states_size});
        CUDNN_CHECK(cudnnSetDropoutDescriptor(
            dropout_desc_,
            ctx()->cudnn_handle(),
            this->ratio(),
            X_states->template mutable_data<uint8_t, Context>(),
            states_size,
            rng_seed_));
      }
    }
    size_t reserve_size;
    CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(input_desc_, &reserve_size));
    auto* X_mask = Buffer("X_mask")->Reshape({(int64_t)reserve_size});
    CUDNN_CHECK(cudnnDropoutForward(
        ctx()->cudnn_handle(),
        dropout_desc_,
        input_desc_,
        X.template data<T, Context>(),
        input_desc_,
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        X_mask->template mutable_data<uint8_t, Context>(),
        reserve_size));
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
void CuDNNDropoutOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNDropoutGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  CuDNNSetTensorDesc<T>(&input_desc_, {dY.count(), 1, 1, 1});
  if (phase() == "TEST") {
    NOT_IMPLEMENTED;
  } else if (phase() == "TRAIN") {
    // Initialize the dropout states
    if (!states_initialized_) {
      states_initialized_ = true;
      size_t states_size;
      CUDNN_CHECK(
          cudnnDropoutGetStatesSize(ctx()->cudnn_handle(), &states_size));
      std::lock_guard<std::mutex> lk(CUDAContext::mutex());
      auto* X_states = workspace()->CreateTensor(
          "/share/cudnn/dropout:" + str::to(rng_seed_) + "/states");
      if (X_states->count() > 0) {
        CUDNN_CHECK(cudnnRestoreDropoutDescriptor(
            dropout_desc_,
            ctx()->cudnn_handle(),
            this->ratio(),
            X_states->template mutable_data<uint8_t, Context>(),
            states_size,
            rng_seed_));
      } else {
        LOG(FATAL) << "Missing dropout states with seed: " << rng_seed_;
      }
    }
    // Check the reserve space
    size_t reserve_size;
    CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(input_desc_, &reserve_size));
    auto* X_mask = Buffer("X_mask");
    CHECK_EQ(X_mask->size(), reserve_size);
    // Compute the gradient using mask
    CUDNN_CHECK(cudnnDropoutBackward(
        ctx()->cudnn_handle(),
        dropout_desc_,
        input_desc_,
        dY.template data<T, Context>(),
        input_desc_,
        dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
        X_mask->template mutable_data<uint8_t, Context>(),
        reserve_size));
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
void CuDNNDropoutGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CUDNN_OPERATOR(Dropout);
DEPLOY_CUDNN_OPERATOR(DropoutGradient);

} // namespace dragon

#endif // CUDNN_VERSION_MIN(7, 0, 0)

#endif // USE_CUDNN
