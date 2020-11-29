#include "dragon/operators/normalization/group_norm_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/filler.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename InputType, typename ParamType>
void GroupNormOp<Context>::DoRunWithType() {
  TENSOR_FILL_WITH_TYPE(Input(1), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(2), vec64_t({C_}), ParamType);

  auto* X_mu = Buffer("X_mu")->Reshape({N_, G_});
  auto* X_rsig = Buffer("X_rsig")->Reshape({N_, G_});
  auto* X_scale = Buffer("X_scale")->Reshape({N_, C_});
  auto* X_bias = Buffer("X_bias")->Reshape({N_, C_});

  auto* x = Input(0).template data<InputType, Context>();
  auto* mu = X_mu->template mutable_data<ParamType, Context>();
  auto* rsig = X_rsig->template mutable_data<ParamType, Context>();

  // Compute the moments
  if (data_format() == "NCHW") {
    vec32_t dims = {(int)(N_ * G_), (int)(D_ * S_)};
    vec32_t axes = {1};
    kernel::Moments(2, dims.data(), 1, axes.data(), x, mu, rsig, ctx());
  } else if (data_format() == "NHWC") {
    vec32_t dims = {(int)N_, (int)S_, (int)G_, (int)D_};
    vec32_t axes = {1, 3};
    kernel::Moments(4, dims.data(), 2, axes.data(), x, mu, rsig, ctx());
  }

  // Inverse stddev from variance
  math::InvStd(N_ * G_, epsilon_, rsig, rsig, ctx());

  // Fuse parameters to compute affine transformation
  kernel::GroupNorm(
      N_,
      G_,
      D_,
      S_,
      data_format(),
      x,
      mu,
      rsig,
      Input(1).template data<ParamType, Context>(), // gamma
      Input(2).template data<ParamType, Context>(), // beta
      X_scale->template mutable_data<ParamType, Context>(),
      X_bias->template mutable_data<ParamType, Context>(),
      Output(0)->template mutable_data<InputType, Context>(),
      ctx());
}

template <class Context>
void GroupNormOp<Context>::RunOnDevice() {
  DetermineBaseArguments();
  Output(0)->ReshapeLike(Input(0));

  if (Input(0).template IsType<float>()) {
    DoRunWithType<float, float>();
  } else if (Input(0).template IsType<float16>()) {
    DoRunWithType<float16, float>();
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float16", "float32"});
  }
}

template <class Context>
template <typename InputType, typename ParamType>
void GroupNormGradientOp<Context>::DoRunWithType() {
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto *X_mu = Buffer("X_mu"), *X_rsig = Buffer("X_rsig");
  auto* X_scale = Buffer("X_scale")->Reshape({N_, G_});
  auto* X_bias = Buffer("X_bias")->Reshape({N_, G_});

  // Gradient w.r.t. gamma, beta and input
  kernel::GroupNormGrad(
      N_,
      G_,
      D_,
      S_,
      data_format(),
      Input(0).template data<InputType, Context>(), // x
      X_mu->template data<ParamType, Context>(),
      X_rsig->template data<ParamType, Context>(),
      Input(1).template data<ParamType, Context>(), // gamma
      Input(2).template data<InputType, Context>(), // dy
      X_scale->template mutable_data<ParamType, Context>(),
      X_bias->template mutable_data<ParamType, Context>(),
      dW->Reshape({C_})->template mutable_data<ParamType, Context>(),
      dB->Reshape({C_})->template mutable_data<ParamType, Context>(),
      dX->template mutable_data<InputType, Context>(),
      ctx());
}

template <class Context>
void GroupNormGradientOp<Context>::RunOnDevice() {
  DetermineBaseArguments();
  Output(0)->ReshapeLike(Input(0));

  if (Input(0).template IsType<float>()) {
    DoRunWithType<float, float>();
  } else if (Input(0).template IsType<float16>()) {
    DoRunWithType<float16, float>();
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float16", "float32"});
  }
}

DEPLOY_CPU_OPERATOR(GroupNorm);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GroupNorm);
#endif

DEPLOY_CPU_OPERATOR(GroupNormGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GroupNormGradient);
#endif

OPERATOR_SCHEMA(GroupNorm)
    /* X, W, B */
    .NumInputs(3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(GroupNormGradient)
    /* X, W, dY */
    .NumInputs(3)
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
        vector<string>({I(0), I(1), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(GroupNorm, GradientMaker);

} // namespace dragon
