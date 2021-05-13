#include "dragon/operators/normalization/group_norm_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void GroupNormOp<Context>::DoRunWithType() {
  using ParamT = typename math::AccmulatorType<T>::type;
  INITIALIZE_TENSOR_VIA_SPEC(Input(1), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(2), vec64_t({C_}), ParamT);
  auto* X_mu = Buffer("X_mu")->Reshape({N_, G_});
  auto* X_rsig = Buffer("X_rsig")->Reshape({N_, G_});

  auto* x = Input(0).template data<T, Context>();
  auto* mu = X_mu->template mutable_data<ParamT, Context>();
  auto* rsig = X_rsig->template mutable_data<ParamT, Context>();

  // Compute the moments
  if (data_format() == "NCHW") {
    vec32_t dims = {int(N_ * G_), int(D_ * S_)};
    vec32_t axes = {1};
    kernels::Moments(2, dims.data(), 1, axes.data(), x, mu, rsig, ctx());
  } else if (data_format() == "NHWC") {
    vec32_t dims = {int(N_), int(S_), int(G_), int(D_)};
    vec32_t axes = {1, 3};
    kernels::Moments(4, dims.data(), 2, axes.data(), x, mu, rsig, ctx());
  }

  // Inverse stddev from variance
  math::InvStd(N_ * G_, epsilon_, rsig, rsig, ctx());

  // Fuse parameters to compute affine transformation
  auto* scratch =
      ctx()->workspace()->template data<ParamT, Context>({2 * N_ * C_})[0];
  kernels::GroupNorm(
      N_,
      G_,
      D_,
      S_,
      data_format(),
      x,
      mu,
      rsig,
      Input(1).template data<ParamT, Context>(), // gamma
      Input(2).template data<ParamT, Context>(), // beta
      scratch,
      scratch + N_ * C_,
      Output(0)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void GroupNormOp<Context>::RunOnDevice() {
  GetBaseArguments();
  Output(0)->ReshapeLike(Input(0));
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void GroupNormGradientOp<Context>::DoRunWithType() {
  using ParamT = typename math::AccmulatorType<T>::type;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto *X_mu = Buffer("X_mu"), *X_rsig = Buffer("X_rsig");

  // Gradient w.r.t. gamma, beta and input
  auto* scratch =
      ctx()->workspace()->template data<ParamT, Context>({2 * N_ * G_})[0];
  kernels::GroupNormGrad(
      N_,
      G_,
      D_,
      S_,
      data_format(),
      Input(0).template data<T, Context>(), // x
      X_mu->template data<ParamT, Context>(),
      X_rsig->template data<ParamT, Context>(),
      Input(1).template data<ParamT, Context>(), // gamma
      Input(2).template data<T, Context>(), // dy
      scratch,
      scratch + N_ * G_,
      dW->Reshape({C_})->template mutable_data<ParamT, Context>(),
      dB->Reshape({C_})->template mutable_data<ParamT, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void GroupNormGradientOp<Context>::RunOnDevice() {
  GetBaseArguments();
  Output(0)->ReshapeLike(Input(0));
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
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
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(GroupNorm, GradientMaker);

} // namespace dragon
