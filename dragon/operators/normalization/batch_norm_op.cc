#include "dragon/operators/normalization/batch_norm_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/filler.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename InputType, typename ParamType>
void BatchNormOp<Context>::TrainingImpl() {
  TENSOR_FILL_WITH_TYPE(Input(1), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(2), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(3), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(4), vec64_t({C_}), ParamType);

  auto* X_mu = Buffer("X_mu")->Reshape({C_});
  auto* X_rsig = Buffer("X_rsig")->Reshape({C_});
  auto* X_scale = Buffer("X_scale")->Reshape({C_});
  auto* X_bias = Buffer("X_bias")->Reshape({C_});

  auto* x = Input(0).template data<InputType, Context>();
  auto* rm = Input(3).template mutable_data<ParamType, Context>();
  auto* rv = Input(4).template mutable_data<ParamType, Context>();
  auto* mu = X_mu->template mutable_data<ParamType, Context>();
  auto* rsig = X_rsig->template mutable_data<ParamType, Context>();
  auto* scale = X_scale->template mutable_data<ParamType, Context>();

  // Compute moments
  if (sync_stats_ > 0) {
#ifdef USE_MPI
    // Compute E(X) and E(X^2)
    kernel::BatchNormExpectation(
        N_,
        C_,
        S_,
        ParamType(1) / (N_ * comm_size_ * S_),
        data_format(),
        x,
        mu,
        rsig,
        ctx());
    // Compute D(X) = E(X^2) - E(X)^2
    ctx()->FinishDeviceComputation();
    if (enable_nccl_) {
#ifdef USE_NCCL
      auto nccl_comm_ = this->nccl_comm();
      auto nccl_dtype_ = this->template nccl_dtype<ParamType>();
      NCCL_CHECK(ncclAllReduce(
          (void*)mu,
          (void*)mu,
          C_,
          nccl_dtype_,
          ncclSum,
          nccl_comm_,
          ((CUDAContext*)ctx())->cuda_stream()));
      NCCL_CHECK(ncclAllReduce(
          (void*)rsig,
          (void*)rsig,
          C_,
          nccl_dtype_,
          ncclSum,
          nccl_comm_,
          ((CUDAContext*)ctx())->cuda_stream()));
#endif // USE_NCCL
    } else {
      AllReduce(mu, mu, C_);
      AllReduce(rsig, rsig, C_);
    }
    math::Square(C_, mu, scale, ctx());
    math::Sub(C_, rsig, scale, rsig, ctx());
#endif // USE_MPI
  } else {
    if (data_format() == "NCHW") {
      vec32_t dims = {(int)N_, (int)C_, (int)S_};
      vec32_t axes = {0, 2};
      kernel::Moments(3, dims.data(), 2, axes.data(), x, mu, rsig, ctx());
    } else if (data_format() == "NHWC") {
      vec32_t dims = {(int)(N_ * S_), (int)C_};
      vec32_t axes = {0};
      kernel::Moments(2, dims.data(), 1, axes.data(), x, mu, rsig, ctx());
    }
  }

  // Compute running statistics
  if (is_recomputing_ == 0) {
    math::Axpby(C_, 1.f - momentum_, mu, momentum_, rm, ctx());
    math::Axpby(C_, 1.f - momentum_, rsig, momentum_, rv, ctx());
  }

  // Inverse stddev from variance
  math::InvStd(C_, epsilon_, rsig, rsig, ctx());

  // Fuse parameters to compute affine transformation
  kernel::BatchNorm(
      N_,
      C_,
      S_,
      data_format(),
      x,
      mu,
      rsig,
      Input(1).template data<ParamType, Context>(), // gamma
      Input(2).template data<ParamType, Context>(), // beta
      scale,
      X_bias->template mutable_data<ParamType, Context>(),
      Output(0)->template mutable_data<InputType, Context>(),
      ctx());
}

template <class Context>
template <typename InputType, typename ParamType>
void BatchNormOp<Context>::InferenceImpl() {
  TENSOR_FILL_WITH_TYPE(Input(1), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(2), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(3), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(4), vec64_t({C_}), ParamType);

  auto* X_rsig = Buffer("X_rsig")->Reshape({C_});
  auto* X_scale = Buffer("X_scale")->Reshape({C_});
  auto* X_bias = Buffer("X_bias")->Reshape({C_});
  auto* rv = Input(4).template data<ParamType, Context>();
  auto* rsig = X_rsig->template mutable_data<ParamType, Context>();

  // Inverse stddev from variance
  math::InvStd(C_, epsilon_, rv, rsig, ctx());

  // Fuse parameters to compute affine transformation
  kernel::BatchNorm(
      N_,
      C_,
      S_,
      data_format(),
      Input(0).template data<InputType, Context>(),
      Input(3).template data<ParamType, Context>(),
      rsig,
      Input(1).template data<ParamType, Context>(), // gamma
      Input(2).template data<ParamType, Context>(), // beta
      X_scale->template mutable_data<ParamType, Context>(),
      X_bias->template mutable_data<ParamType, Context>(),
      Output(0)->template mutable_data<InputType, Context>(),
      ctx());
}

template <class Context>
void BatchNormOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Get the recomputing flag
  auto* flag = workspace()->GetTensor("/share/flag/recomputing");
  is_recomputing_ = flag->template data<bool, CPUContext>()[0] ? 1 : 0;

  // Dispatch the training or inference impl
  Output(0)->ReshapeLike(Input(0));
  if (Input(0).template IsType<float>()) {
    if (is_training_) {
      TrainingImpl<float, float>();
    } else {
      InferenceImpl<float, float>();
    }
  } else if (Input(0).template IsType<float16>()) {
    if (is_training_) {
      TrainingImpl<float16, float>();
    } else {
      InferenceImpl<float16, float>();
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float16", "float32"});
  }
}

template <class Context>
template <typename InputType, typename ParamType>
void BatchNormGradientOp<Context>::TrainingImpl() {
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto *X_mu = Buffer("X_mu"), *X_rsig = Buffer("X_rsig");
  auto *X_scale = Buffer("X_scale"), *X_bias = Buffer("X_bias");

  auto* x = Input(0).template data<InputType, Context>();
  auto* gamma = Input(1).template data<ParamType, Context>();
  auto* dy = Input(4).template data<InputType, Context>();
  auto* mu = X_mu->template data<ParamType, Context>();
  auto* rsig = X_rsig->template data<ParamType, Context>();
  auto* scale = X_scale->template mutable_data<ParamType, Context>();
  auto* bias = X_bias->template mutable_data<ParamType, Context>();
  auto* dgamma = dW->Reshape({C_})->template mutable_data<ParamType, Context>();
  auto* dbeta = dB->Reshape({C_})->template mutable_data<ParamType, Context>();

  // Gradient w.r.t. gamma and beta
  kernel::BatchNormInternalGrad(
      N_, C_, S_, data_format(), x, mu, rsig, gamma, dy, dgamma, dbeta, ctx());

  if (sync_stats_ > 0) {
#ifdef USE_MPI
    ctx()->FinishDeviceComputation();
    if (enable_nccl_) {
#ifdef USE_NCCL
      auto nccl_comm_ = this->nccl_comm();
      auto nccl_dtype_ = this->template nccl_dtype<ParamType>();
      NCCL_CHECK(ncclAllReduce(
          (void*)dgamma,
          (void*)scale,
          C_,
          nccl_dtype_,
          ncclSum,
          nccl_comm_,
          ((CUDAContext*)ctx())->cuda_stream()));
      NCCL_CHECK(ncclAllReduce(
          (void*)dbeta,
          (void*)bias,
          C_,
          nccl_dtype_,
          ncclSum,
          nccl_comm_,
          ((CUDAContext*)ctx())->cuda_stream()));
#endif // USE_NCCL
    } else {
      AllReduce(dgamma, scale, C_);
      AllReduce(dbeta, bias, C_);
    }
    math::Scale(C_, ParamType(1) / comm_size_, scale, scale, ctx());
    math::Scale(C_, ParamType(1) / comm_size_, bias, bias, ctx());
#endif // USE_MPI
  } else {
    scale = dgamma, bias = dbeta;
  }

  // Gradient w.r.t. input
  kernel::BatchNormTrainingGrad(
      N_,
      C_,
      S_,
      data_format(),
      x,
      mu,
      rsig,
      gamma,
      scale,
      bias,
      dy,
      Output(0)->template mutable_data<InputType, Context>(),
      ctx());
}

template <class Context>
template <typename InputType, typename ParamType>
void BatchNormGradientOp<Context>::InferenceImpl() {
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto* X_scale = Buffer("X_scale")->Reshape({C_});

  auto* rv = Input(3).template data<ParamType, Context>();
  auto* rsig = X_scale->template mutable_data<ParamType, Context>();

  // Gradient w.r.t. gamma or beta if necessary
  ParamType *dgamma = nullptr, *dbeta = nullptr;
  if (dW->has_name() || dB->has_name()) {
    dgamma = dW->Reshape({C_})->template mutable_data<ParamType, Context>();
    dbeta = dB->Reshape({C_})->template mutable_data<ParamType, Context>();
  }

  // Inverse stddev from variance
  math::InvStd(C_, epsilon_, rv, rsig, ctx());

  // Gradient w.r.t. gamma, beta and input
  kernel::BatchNormInferenceGrad(
      N_,
      C_,
      S_,
      data_format(),
      Input(0).template data<InputType, Context>(), // x
      Input(2).template data<ParamType, Context>(), // rm
      rsig,
      Input(1).template data<ParamType, Context>(), // gamma
      Input(4).template data<InputType, Context>(), // dy
      dgamma,
      dbeta,
      dX->template mutable_data<InputType, Context>(),
      ctx());
}

template <class Context>
void BatchNormGradientOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Dispatch the training or inference impl
  Output(0)->ReshapeLike(Input(0));
  if (Input(0).template IsType<float>()) {
    if (is_training_ > 0) {
      TrainingImpl<float, float>();
    } else {
      InferenceImpl<float, float>();
    }
  } else if (Input(0).template IsType<float16>()) {
    if (is_training_ > 0) {
      TrainingImpl<float16, float>();
    } else {
      InferenceImpl<float16, float>();
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float16", "float32"});
  }
}

DEPLOY_CPU_OPERATOR(BatchNorm);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(BatchNorm);
#endif

DEPLOY_CPU_OPERATOR(BatchNormGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(BatchNormGradient);
#endif

OPERATOR_SCHEMA(BatchNorm)
    /* X, W, B, RunningMean, RunningVar */
    .NumInputs(5)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(BatchNormGradient)
    /* X, W, RunningMean, RunningVar, dY */
    .NumInputs(5)
    /* dX, dW, dB */
    .NumOutputs(3);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(0), I(1), I(3), I(4), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(BatchNorm, GradientMaker);

} // namespace dragon
