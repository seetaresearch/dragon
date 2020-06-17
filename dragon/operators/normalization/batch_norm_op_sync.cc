#ifdef USE_MPI

#include "dragon/core/workspace.h"
#include "dragon/operators/normalization/batch_norm_op.h"
#include "dragon/utils/filler.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename InputType, typename ParamType>
void SyncBatchNormOp<Context>::TrainingImpl() {
  TENSOR_FILL_WITH_TYPE(Input(1), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(2), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(3), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(4), vec64_t({C_}), ParamType);

  auto* X_mu = Buffer("X_mu")->Reshape({C_});
  auto* X_rsig = Buffer("X_rsig")->Reshape({C_});
  auto* X_scale = Buffer("X_scale")->Reshape({C_});
  auto* X_bias = Buffer("X_bias")->Reshape({C_});

  auto* x = Input(0).template data<InputType, Context>();
  auto* gamma = Input(1).template data<ParamType, Context>();
  auto* beta = Input(2).template data<ParamType, Context>();
  auto* rm = Input(3).template mutable_data<ParamType, Context>();
  auto* rv = Input(4).template mutable_data<ParamType, Context>();
  auto* mu = X_mu->template mutable_data<ParamType, Context>();
  auto* rsig = X_rsig->template mutable_data<ParamType, Context>();
  auto* scale = X_scale->template mutable_data<ParamType, Context>();
  auto* bias = X_bias->template mutable_data<ParamType, Context>();
  auto* y = Output(0)->template mutable_data<InputType, Context>();

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
#endif
  } else {
    AllReduce(mu, mu, C_);
    AllReduce(rsig, rsig, C_);
  }
  math::Square(C_, mu, y, ctx());
  math::Sub(C_, rsig, y, rsig, ctx());

  // Compute running statistics
  if (is_recomputing_ == 0) {
    // Running(X) = (1 - momentum) * Cur(X) + momentum * Running(X)
    math::Axpby(C_, 1.f - momentum_, mu, momentum_, rm, ctx());
    math::Axpby(C_, 1.f - momentum_, rsig, momentum_, rv, ctx());
  }

  // Fuse parameters along channel axis
  // [mu, rsig, alpha, beta] => [scale, bias]
  math::InvStd(C_, eps_, rsig, rsig, ctx());
  math::Mul(C_, gamma, rsig, scale, ctx());
  math::Mul(C_, scale, mu, bias, ctx());
  math::Sub(C_, beta, bias, bias, ctx());

  // Compute affine transformation
  if (data_format() == "NCHW") {
    kernel::Affine(N_, C_, S_, x, scale, bias, y, ctx());
  } else if (data_format() == "NHWC") {
    kernel::Affine(N_ * S_, C_, 1, x, scale, bias, y, ctx());
  }
}

template <class Context>
void SyncBatchNormOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Get the recomputing flag
  auto* flag = ws()->GetTensor("/share/flag/recomputing");
  is_recomputing_ = flag->template data<bool, CPUContext>()[0];

  // Dispatch the training or inference impl
  Output(0)->ReshapeLike(Input(0));
  if (XIsType(Input(0), float)) {
    if (is_training_ > 0) {
      TrainingImpl<float, float>();
    } else {
      this->template InferenceImpl<float, float>();
    }
  } else {
    LOG(FATAL) << TypeString(Input(0), {"float32"});
  }
}

template <class Context>
template <typename InputType, typename ParamType>
void SyncBatchNormGradientOp<Context>::TrainingImpl() {
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

  // Gradient w.r.t. gamma and beta of local batch
  kernel::BatchNormInternalGrad(
      N_, C_, S_, data_format(), x, mu, rsig, gamma, dy, dgamma, dbeta, ctx());

  // Gradient w.r.t. gamma and beta of global batch
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
#endif
  } else {
    AllReduce(dgamma, scale, C_);
    AllReduce(dbeta, bias, C_);
  }
  math::Scale(C_, ParamType(1) / comm_size_, scale, scale, ctx());
  math::Scale(C_, ParamType(1) / comm_size_, bias, bias, ctx());

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
void SyncBatchNormGradientOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Dispatch the training or inference impl
  Output(0)->ReshapeLike(Input(0));
  if (XIsType(Input(0), float)) {
    if (is_training_ > 0) {
      TrainingImpl<float, float>();
    } else {
      this->template InferenceImpl<float, float>();
    }
  } else {
    LOG(FATAL) << TypeString(Input(0), {"float32"});
  }
}

DEPLOY_CPU(SyncBatchNorm);
#ifdef USE_CUDA
DEPLOY_CUDA(SyncBatchNorm);
#endif

DEPLOY_CPU(SyncBatchNormGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(SyncBatchNormGradient);
#endif

OPERATOR_SCHEMA(SyncBatchNorm)
    /* X, W, B, RunningMean, RunningVar */
    .NumInputs(5)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SyncBatchNormGradient)
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

REGISTER_GRADIENT(SyncBatchNorm, GradientMaker);

} // namespace dragon

#endif // USE_MPI
