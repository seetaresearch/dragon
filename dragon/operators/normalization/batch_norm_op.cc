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
  auto* gamma = Input(1).template data<ParamType, Context>();
  auto* beta = Input(2).template data<ParamType, Context>();
  auto* rm = Input(3).template mutable_data<ParamType, Context>();
  auto* rv = Input(4).template mutable_data<ParamType, Context>();
  auto* mu = X_mu->template mutable_data<ParamType, Context>();
  auto* rsig = X_rsig->template mutable_data<ParamType, Context>();
  auto* scale = X_scale->template mutable_data<ParamType, Context>();
  auto* bias = X_bias->template mutable_data<ParamType, Context>();
  auto* y = Output(0)->template mutable_data<InputType, Context>();

  // Compute moments
  if (data_format() == "NCHW") {
    vec32_t dims = {(int)N_, (int)C_, (int)S_};
    vec32_t axes = {0, 2};
    kernel::Moments(3, dims.data(), 2, axes.data(), x, mu, rsig, ctx());
  } else if (data_format() == "NHWC") {
    vec32_t dims = {(int)(N_ * S_), (int)C_};
    vec32_t axes = {0};
    kernel::Moments(2, dims.data(), 1, axes.data(), x, mu, rsig, ctx());
  }

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
template <typename InputType, typename ParamType>
void BatchNormOp<Context>::InferenceImpl() {
  TENSOR_FILL_WITH_TYPE(Input(1), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(2), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(3), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(4), vec64_t({C_}), ParamType);

  auto* X_scale = Buffer("X_scale")->Reshape({C_});
  auto* X_bias = Buffer("X_bias")->Reshape({C_});

  auto* x = Input(0).template data<InputType, Context>();
  auto* gamma = Input(1).template data<ParamType, Context>();
  auto* beta = Input(2).template data<ParamType, Context>();
  auto* rm = Input(3).template data<ParamType, Context>();
  auto* rv = Input(4).template data<ParamType, Context>();
  auto* scale = X_scale->template mutable_data<ParamType, Context>();
  auto* bias = X_bias->template mutable_data<ParamType, Context>();
  auto* y = Output(0)->template mutable_data<InputType, Context>();

  // Fuse parameters along channel axis
  // [mu, rsig, alpha, beta] => [scale, bias]
  math::InvStd(C_, eps_, rv, bias, ctx());
  math::Mul(C_, gamma, bias, scale, ctx());
  math::Mul(C_, scale, rm, bias, ctx());
  math::Sub(C_, beta, bias, bias, ctx());

  // Compute affine transformation
  if (data_format() == "NCHW") {
    kernel::Affine(N_, C_, S_, x, scale, bias, y, ctx());
  } else if (data_format() == "NHWC") {
    kernel::Affine(N_ * S_, C_, 1, x, scale, bias, y, ctx());
  }
}

template <class Context>
void BatchNormOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Get the recomputing flag
  auto* flag = ws()->GetTensor("/share/flag/recomputing");
  is_recomputing_ = flag->template data<bool, CPUContext>()[0];

  // Dispatch the training or inference impl
  Output(0)->ReshapeLike(Input(0));
  if (XIsType(Input(0), float)) {
    if (is_training_) {
      TrainingImpl<float, float>();
    } else {
      InferenceImpl<float, float>();
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float32"});
  }
}

template <class Context>
template <typename InputType, typename ParamType>
void BatchNormGradientOp<Context>::TrainingImpl() {
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto *X_mu = Buffer("X_mu"), *X_rsig = Buffer("X_rsig");

  // Gradient w.r.t. gamma, beta and input
  kernel::BatchNormBackwardTraining(
      N_,
      C_,
      S_,
      data_format(),
      Input(0).template data<InputType, Context>(), // x
      X_mu->template data<ParamType, Context>(), // mu
      X_rsig->template data<ParamType, Context>(), // rsig
      Input(1).template data<ParamType, Context>(), // gamma
      Input(4).template data<InputType, Context>(), // dy
      Output(0)->template mutable_data<InputType, Context>(), // dx
      dW->Reshape({C_})->template mutable_data<ParamType, Context>(), // dgamma
      dB->Reshape({C_})->template mutable_data<ParamType, Context>(), // dbeta
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

  // Restore inverse stddev from variance
  math::InvStd(C_, eps_, rv, rsig, ctx());

  // Gradient w.r.t. gamma, beta and input
  kernel::BatchNormBackwardInference(
      N_,
      C_,
      S_,
      data_format(),
      Input(0).template data<InputType, Context>(), // x
      Input(2).template data<ParamType, Context>(), // rm
      rsig,
      Input(1).template data<ParamType, Context>(), // gamma
      Input(4).template data<InputType, Context>(), // dy
      dX->template mutable_data<InputType, Context>(),
      dgamma,
      dbeta,
      ctx());
}

template <class Context>
void BatchNormGradientOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Dispatch the training or inference impl
  Output(0)->ReshapeLike(Input(0));
  if (XIsType(Input(0), float)) {
    if (is_training_ > 0) {
      TrainingImpl<float, float>();
    } else {
      InferenceImpl<float, float>();
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float32"});
  }
}

DEPLOY_CPU(BatchNorm);
#ifdef USE_CUDA
DEPLOY_CUDA(BatchNorm);
#endif

DEPLOY_CPU(BatchNormGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(BatchNormGradient);
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
