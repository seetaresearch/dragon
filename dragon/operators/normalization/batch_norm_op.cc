#include "dragon/operators/normalization/batch_norm_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void BatchNormOp<Context>::RunTraining() {
  using AccT = typename math::Traits<T>::accumulator_type;
  INITIALIZE_TENSOR_VIA_SPEC(Input(1), vec64_t({C_}), AccT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(2), vec64_t({C_}), AccT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(3), vec64_t({C_}), AccT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(4), vec64_t({C_}), AccT);

  auto* X_mu = Output("X_mu")->Reshape({C_});
  auto* X_rsig = Output("X_rsig")->Reshape({C_});
  auto* X_params = Output("X_params")->Reshape({C_ * 2});

  auto* x = Input(0).template data<T, Context>();
  auto* rm = Input(3).template mutable_data<AccT, Context>();
  auto* rv = Input(4).template mutable_data<AccT, Context>();
  auto* mu = X_mu->template mutable_data<AccT, Context>();
  auto* rsig = X_rsig->template mutable_data<AccT, Context>();
  auto* params = X_params->template mutable_data<AccT, Context>();

  // Compute moments.
  if (sync_stats_ > 0) {
    int64_t N = N_;
    coll_impl_.AllReduce(&N, &N, 1);
    // Compute E(X) and E(X^2)
    kernels::BatchNormExpectation(
        N_,
        C_,
        S_,
        float(N * S_),
        data_format(),
        x,
        params,
        params + C_,
        ctx());
    ctx()->FinishDeviceComputation();
    coll_impl_.AllReduce(params, params, C_ * 2, ctx());
    // Compute D(X) = E(X^2) - E(X)^2
    math::Copy(C_, params, mu, ctx());
    math::Square(C_, params, params, ctx());
    math::Sub(C_, params + C_, params, rsig, ctx());
  } else {
    if (data_format() == "NCHW") {
      vec64_t dims = {N_, C_, S_};
      vec64_t axes = {0, 2};
      kernels::Moments(3, dims.data(), 2, axes.data(), x, mu, rsig, ctx());
    } else if (data_format() == "NHWC") {
      vec64_t dims = {N_ * S_, C_};
      vec64_t axes = {0};
      kernels::Moments(2, dims.data(), 1, axes.data(), x, mu, rsig, ctx());
    }
  }

  // Compute running stats.
  const float decay = momentum();
  math::Axpby(C_, 1.f - decay, mu, decay, rm, ctx());
  math::Axpby(C_, 1.f - decay, rsig, decay, rv, ctx());

  // Compute stddev from variance.
  math::InvStd(C_, epsilon_, rsig, rsig, ctx());

  // Fuse parameters to compute affine transformation.
  kernels::BatchNorm(
      N_,
      C_,
      S_,
      data_format(),
      x,
      mu,
      rsig,
      Input(1).template data<AccT, Context>(), // gamma
      Input(2).template data<AccT, Context>(), // beta
      params,
      params, /* params + C_ */
      Output(0)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void BatchNormOp<Context>::RunInference() {
  using AccT = typename math::Traits<T>::accumulator_type;
  INITIALIZE_TENSOR_VIA_SPEC(Input(1), vec64_t({C_}), AccT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(2), vec64_t({C_}), AccT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(3), vec64_t({C_}), AccT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(4), vec64_t({C_}), AccT);

  auto* X_params = Output("X_params")->Reshape({C_ * 3});
  auto* rv = Input(4).template data<AccT, Context>();
  auto* params = X_params->template mutable_data<AccT, Context>();

  // Compute stddev from variance.
  math::InvStd(C_, epsilon_, rv, params, ctx());

  // Fuse parameters to compute affine transformation.
  kernels::BatchNorm(
      N_,
      C_,
      S_,
      data_format(),
      Input(0).template data<T, Context>(),
      Input(3).template data<AccT, Context>(),
      params,
      Input(1).template data<AccT, Context>(), // gamma
      Input(2).template data<AccT, Context>(), // beta
      params, /* params + C_ */
      params, /* params + C_ * 2 */
      Output(0)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void BatchNormGradientOp<Context>::RunTraining() {
  using AccT = typename math::Traits<T>::accumulator_type;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto &X_mu = Input("X_mu"), &X_rsig = Input("X_rsig");
  auto* X_params = Output("X_params")->Reshape({C_ * 2});

  auto* x = Input(0).template data<T, Context>();
  auto* gamma = Input(1).template data<AccT, Context>();
  auto* dy = Input(4).template data<T, Context>();
  auto* mu = X_mu.template data<AccT, Context>();
  auto* rsig = X_rsig.template data<AccT, Context>();
  auto* params = X_params->template mutable_data<AccT, Context>();

  // Gradient w.r.t. affine transformation.
  kernels::BatchNormWGrad(
      N_,
      C_,
      S_,
      data_format(),
      x,
      mu,
      rsig,
      dy,
      params,
      params, /* params + C_ */
      ctx());

  // Gradient w.r.t. gamma.
  if (dW->has_name()) {
    math::Copy(
        C_,
        params,
        dW->Reshape({C_})->template mutable_data<AccT, Context>(),
        ctx());
  }

  // Gradient w.r.t. beta.
  if (dB->has_name()) {
    math::Copy(
        C_,
        C_, // x_offset
        0, // y_offset
        params,
        dB->Reshape({C_})->template mutable_data<AccT, Context>(),
        ctx());
  }

  int64_t N = N_; // Total batch size.
  if (sync_stats_ > 0) {
    coll_impl_.AllReduce(&N, &N, 1);
    ctx()->FinishDeviceComputation();
    coll_impl_.AllReduce(params, params, C_ * 2, ctx());
  }

  // Gradient w.r.t. input.
  if (dX->has_name()) {
    kernels::BatchNormTrainingGrad(
        N_,
        C_,
        S_,
        float(N * S_),
        data_format(),
        x,
        mu,
        rsig,
        gamma,
        params,
        params, /* params + C_ */
        dy,
        dX->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void BatchNormGradientOp<Context>::RunInference() {
  using AccT = typename math::Traits<T>::accumulator_type;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto* X_params = Output("X_params")->Reshape({C_});

  auto* rv = Input(3).template data<AccT, Context>();
  auto* params = X_params->template mutable_data<AccT, Context>();
  AccT *dgamma = nullptr, *dbeta = nullptr;
  if (dW->has_name() || dB->has_name()) {
    dgamma = dW->Reshape({C_})->template mutable_data<AccT, Context>();
    dbeta = dB->Reshape({C_})->template mutable_data<AccT, Context>();
  }

  // Compute stddev from variance.
  math::InvStd(C_, epsilon_, rv, params, ctx());

  // Gradient w.r.t. gamma, beta and input.
  kernels::BatchNormInferenceGrad(
      N_,
      C_,
      S_,
      data_format(),
      Input(0).template data<T, Context>(), // x
      Input(2).template data<AccT, Context>(), // mu
      params, // rsig
      Input(1).template data<AccT, Context>(), // gamma
      Input(4).template data<T, Context>(), // dy
      dgamma,
      dbeta,
      dX->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(BatchNorm);
DEPLOY_CPU_OPERATOR(BatchNormGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(BatchNorm);
DEPLOY_CUDA_OPERATOR(BatchNormGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(BatchNorm, BatchNorm);
DEPLOY_MPS_OPERATOR(BatchNormGradient, BatchNormGradient);
#endif
DEFINE_OP_SINGLE_ARG(float, BatchNormOp, momentum);

OPERATOR_SCHEMA(BatchNorm)
    /* X, W, B, M, V */
    .NumInputs(5)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(BatchNormGradient)
    /* X, W, M, V, dY */
    .NumInputs(5)
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
        vector<string>({I(0), I(1), I(3), I(4), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(BatchNorm, GradientMaker);

} // namespace dragon
