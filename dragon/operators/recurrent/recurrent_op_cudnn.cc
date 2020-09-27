#ifdef USE_CUDNN

#include "dragon/operators/recurrent/recurrent_op_cudnn.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/filler.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNRecurrentOpBase<Context>::ResetDesc() {
  input_dims_ = Input(0).dims();
  seq_length_ = Input(0).dim(0);
  auto input_type = TypeMeta::Id<T>();
  auto batch_size = Input(0).dim(1);
  auto x_dim = Input(0).dim(2);
  auto ndirections = bidirectional_ ? 2 : 1;
  auto y_dim = hidden_size_ * ndirections;

  // Setup Dropout
  if (dropout_ratio_ < 1.f) {
#if CUDNN_VERSION_MIN(7, 0, 0)
    if (!states_initialized_) {
      states_initialized_ = 1;
      CUDNN_CHECK(
          cudnnDropoutGetStatesSize(ctx()->cudnn_handle(), &states_size_));
      std::lock_guard<std::mutex> lk(CUDAContext::mutex());
      auto* states_tensor = workspace()->CreateTensor(
          "/share/cudnn/dropout:" + str::to(rng_seed_) + "/states");
      if (states_tensor->count() > 0) {
        auto* states = states_tensor->template mutable_data<uint8_t, Context>();
        CUDNN_CHECK(cudnnRestoreDropoutDescriptor(
            dropout_desc_,
            ctx()->cudnn_handle(),
            dropout_ratio_,
            states,
            states_size_,
            rng_seed_));
      } else {
        auto* states = states_tensor->Reshape({(int64_t)states_size_})
                           ->template mutable_data<uint8_t, Context>();
        CUDNN_CHECK(cudnnSetDropoutDescriptor(
            dropout_desc_,
            ctx()->cudnn_handle(),
            dropout_ratio_,
            states,
            states_size_,
            rng_seed_));
      }
    }
#else
    LOG(FATAL) << "Dropout has been supported since CuDNN 7.0";
#endif
  }

  // Setup RNN
  if (input_type == TypeMeta::Id<float16>()) {
    compute_type_ = CUDNN_DATA_FLOAT;
  } else if (input_type == TypeMeta::Id<float>()) {
    compute_type_ = CUDNN_DATA_FLOAT;
  } else if (input_type == TypeMeta::Id<double>()) {
    compute_type_ = CUDNN_DATA_DOUBLE;
  }
#if CUDNN_VERSION_MIN(7, 0, 0)
  CUDNN_CHECK(cudnnSetRNNDescriptor_v6(
      ctx()->cudnn_handle(),
      rnn_desc_,
      hidden_size_,
      num_layers_,
      dropout_desc_,
      rnn_input_mode_,
      rnn_direction_,
      rnn_mode_,
      CUDNN_RNN_ALGO_STANDARD,
      compute_type_));
#else
  CUDNN_CHECK(cudnnSetRNNDescriptor(
      rnn_desc_,
      hidden_size_,
      num_layers_,
      dropout_desc_,
      rnn_input_mode_,
      rnn_direction_,
      rnn_mode_,
      compute_type_));
#endif

  // Setup TensorCore
#if CUDNN_VERSION_MIN(7, 0, 0)
  if (enable_tensor_core_ > 0) {
    cudnnMathType_t math_type;
    if (input_type == TypeMeta::Id<float16>()) {
      math_type = CUDNN_TENSOR_OP_MATH;
    } else {
      math_type = CUDNN_DEFAULT_MATH;
#if CUDNN_VERSION_MIN(8, 0, 0)
      if (!CUDAContext::objects().cudnn_allow_tf32_) {
        math_type = CUDNN_FMA_MATH;
      }
#endif
    }
    CUDNN_CHECK(cudnnSetRNNMatrixMathType(rnn_desc_, math_type));
  }
#endif

  // Setup X and Y
  output_dims_ = {seq_length_, batch_size, y_dim};
  x_descs_.reset(new CuDNNTensorDescs(seq_length_));
  y_descs_.reset(new CuDNNTensorDescs(seq_length_));
  x_descs_->Set<T>({batch_size, x_dim, 1}, {x_dim, 1, 1});
  y_descs_->Set<T>({batch_size, y_dim, 1}, {y_dim, 1, 1});

  // Setup Hx, Cx, Hy and Cy
  hidden_dims_ = {num_layers_ * ndirections, batch_size, hidden_size_};
  CuDNNSetTensorDesc<T>(&hx_desc_, hidden_dims_);
  CuDNNSetTensorDesc<T>(&cx_desc_, hidden_dims_);
  CuDNNSetTensorDesc<T>(&hy_desc_, hidden_dims_);
  CuDNNSetTensorDesc<T>(&cy_desc_, hidden_dims_);

  // Setup packed weights
  size_t w_size;
  CUDNN_CHECK(cudnnGetRNNParamsSize(
      ctx()->cudnn_handle(),
      rnn_desc_,
      x_descs_->data()[0],
      &w_size,
      CuDNNType<T>::type));
  int64_t w_count = (int64_t)w_size / sizeof(T);
  CHECK_EQ(w_count, Input(1).count())
      << "\nModel request "
      << "Tensor(" << Input(1).name() << ")'s "
      << "size is " << w_count << ", \n"
      << "but now is " << Input(1).count() << ", "
      << "did you feed the incorrect data before?";
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(
      w_desc_,
      CuDNNType<T>::type,
      CUDNN_TENSOR_NCHW,
      3,
      vec32_t({(int)w_count, 1, 1}).data()));

  // Determine the RNN workspace
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
      ctx()->cudnn_handle(),
      rnn_desc_,
      seq_length_,
      x_descs_->data(),
      &workspace_size_));
}

template <class Context>
template <typename T>
void CuDNNRecurrentOp<Context>::DoRunWithType() {
  if (Input(0).dims() != input_dims_) {
    this->template ResetDesc<T>();
  }

  if (InputSize() > 2) {
    TENSOR_FILL(Input(2), hidden_dims_);
  }
  if (InputSize() > 3) {
    TENSOR_FILL(Input(3), hidden_dims_);
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

  auto* scratch =
      ctx()->workspace()->template data<Context>({workspace_size_})[0];

  if (phase() == "TRAIN") {
    CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seq_length_,
        x_descs_->data(),
        &reserve_size_));
    auto* reserve = Buffer("reserve")
                        ->Reshape({(int64_t)reserve_size_})
                        ->template mutable_data<uint8_t, Context>();
    CUDNN_CHECK(cudnnRNNForwardTraining(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seq_length_,
        x_descs_->data(),
        xAt(0),
        hx_desc_,
        xAt(2),
        cx_desc_,
        xAt(3),
        w_desc_,
        xAt(1),
        y_descs_->data(),
        yAt(0),
        hy_desc_,
        yAt(1),
        cy_desc_,
        yAt(2),
        scratch,
        workspace_size_,
        reserve,
        reserve_size_));
  } else if (phase() == "TEST") {
    CUDNN_CHECK(cudnnRNNForwardInference(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seq_length_,
        x_descs_->data(),
        xAt(0),
        hx_desc_,
        xAt(2),
        cx_desc_,
        xAt(3),
        w_desc_,
        xAt(1),
        y_descs_->data(),
        yAt(0),
        hy_desc_,
        yAt(1),
        cy_desc_,
        yAt(2),
        scratch,
        workspace_size_));
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
void CuDNNRecurrentOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float, float16>>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNRecurrentGradientOp<Context>::DoRunWithType() {
  if (Input(0).dims() != input_dims_) {
    this->template ResetDesc<T>();
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

  auto* scratch =
      ctx()->workspace()->template data<Context>({workspace_size_})[0];

  // Check the ReserveSpace
  CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
      ctx()->cudnn_handle(),
      rnn_desc_,
      seq_length_,
      x_descs_->data(),
      &reserve_size_));
  auto* reserve_tensor = Buffer("reserve");
  CHECK_EQ(reserve_size_, reserve_tensor->nbytes());
#if CUDNN_VERSION_MIN(6, 0, 0)
  auto* reserve = reserve_tensor->template mutable_data<uint8_t, Context>();
#else
  auto* reserve = reserve_tensor->template data<uint8_t, Context>();
#endif

  if (Output(0)->has_name() || Output(1)->has_name() || Output(2)->has_name() ||
      Output(3)->has_name()) {
    CUDNN_CHECK(cudnnRNNBackwardData(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seq_length_,
        y_descs_->data(),
        xAt(4), // Y
        y_descs_->data(),
        xAt(5), // dY
        hy_desc_,
        xAt(6), // dHy
        cy_desc_,
        xAt(7), // dCy
        w_desc_,
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
        scratch,
        workspace_size_,
        reserve,
        reserve_size_));
  }

  if (Output(1)->has_name()) {
    // CuDNN accumulates the gradient of weights
    // We should reset them before bakcward computing
    math::Set(Output(1)->count(), cast::to<T>(0.f), yAt(1), ctx());
    CUDNN_CHECK(cudnnRNNBackwardWeights(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seq_length_,
        x_descs_->data(),
        xAt(0), // X
        hx_desc_,
        xAt(2), // Hx
        y_descs_->data(),
        xAt(4), // Y
        scratch,
        workspace_size_,
        w_desc_,
        yAt(1), // dW
        reserve,
        reserve_size_));
  }
}

template <class Context>
void CuDNNRecurrentGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(0)); // dX
  Output(1)->ReshapeLike(Input(1)); // dW
  Output(2)->ReshapeLike(Input(2)); // dHx
  Output(3)->ReshapeLike(Input(3)); // dCx
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CUDNN_OPERATOR(Recurrent);
DEPLOY_CUDNN_OPERATOR(RecurrentGradient);

} // namespace dragon

#endif // USE_CUDNN
