#include "dragon/operators/normalization/batch_norm_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void BatchNormOp<Context>::RunTraining() {
  using ParamT = typename math::AccumulatorType<T>::type;
  INITIALIZE_TENSOR_VIA_SPEC(Input(1), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(2), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(3), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(4), vec64_t({C_}), ParamT);

  auto* X_mu = Output("X_mu")->Reshape({C_});
  auto* X_rsig = Output("X_rsig")->Reshape({C_});
  auto* X_params = Output("X_params")->Reshape({C_ * 2});

  auto* x = Input(0).template data<T, Context>();
  auto* rm = Input(3).template mutable_data<ParamT, Context>();
  auto* rv = Input(4).template mutable_data<ParamT, Context>();
  auto* mu = X_mu->template mutable_data<ParamT, Context>();
  auto* rsig = X_rsig->template mutable_data<ParamT, Context>();
  auto* params = X_params->template mutable_data<ParamT, Context>();

  // Compute moments.
  if (sync_stats_ > 0) {
#ifdef USE_MPI
    // Compute E(X) and E(X^2)
    kernels::BatchNormExpectation(
        N_,
        C_,
        S_,
        float(N_ * S_ * comm_size_),
        data_format(),
        x,
        params,
        params + C_,
        ctx());
    ctx()->FinishDeviceComputation();
    if (enable_nccl_) {
#ifdef USE_NCCL
      NCCL_CHECK(ncclAllReduce(
          (void*)params,
          (void*)params,
          C_ * 2,
          this->template nccl_data_type<ParamT>(),
          ncclSum,
          this->nccl_comm(),
          ((CUDAContext*)ctx())->cuda_stream()));
#endif // USE_NCCL
    } else {
      AllReduce(params, params, C_ * 2);
    }
    // Compute D(X) = E(X^2) - E(X)^2
    math::Copy(C_, params, mu, ctx());
    math::Square(C_, params, params, ctx());
    math::Sub(C_, params + C_, params, rsig, ctx());
#endif // USE_MPI
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
      Input(1).template data<ParamT, Context>(), // gamma
      Input(2).template data<ParamT, Context>(), // beta
      params,
      params + C_,
      Output(0)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void BatchNormOp<Context>::RunInference() {
  using ParamT = typename math::AccumulatorType<T>::type;
  INITIALIZE_TENSOR_VIA_SPEC(Input(1), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(2), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(3), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(4), vec64_t({C_}), ParamT);

  auto* X_params = Output("X_params")->Reshape({C_ * 3});
  auto* rv = Input(4).template data<ParamT, Context>();
  auto* params = X_params->template mutable_data<ParamT, Context>();

  // Compute stddev from variance.
  math::InvStd(C_, epsilon_, rv, params, ctx());

  // Fuse parameters to compute affine transformation.
  kernels::BatchNorm(
      N_,
      C_,
      S_,
      data_format(),
      Input(0).template data<T, Context>(),
      Input(3).template data<ParamT, Context>(),
      params,
      Input(1).template data<ParamT, Context>(), // gamma
      Input(2).template data<ParamT, Context>(), // beta
      params + C_,
      params + C_ * 2,
      Output(0)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void BatchNormGradientOp<Context>::RunTraining() {
  using ParamT = typename math::AccumulatorType<T>::type;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto &X_mu = Input("X_mu"), &X_rsig = Input("X_rsig");
  auto* X_params = Output("X_params")->Reshape({C_ * 2});

  auto* x = Input(0).template data<T, Context>();
  auto* gamma = Input(1).template data<ParamT, Context>();
  auto* dy = Input(4).template data<T, Context>();
  auto* mu = X_mu.template data<ParamT, Context>();
  auto* rsig = X_rsig.template data<ParamT, Context>();
  auto* params = X_params->template mutable_data<ParamT, Context>();

  // Gradient w.r.t. affine transformation.
  kernels::BatchNormWGrad(
      N_, C_, S_, data_format(), x, mu, rsig, dy, params, params + C_, ctx());

  // Gradient w.r.t. gamma.
  if (dW->has_name()) {
    math::Copy(
        C_,
        params,
        dW->Reshape({C_})->template mutable_data<ParamT, Context>(),
        ctx());
  }

  // Gradient w.r.t. beta.
  if (dB->has_name()) {
    math::Copy(
        C_,
        params + C_,
        dB->Reshape({C_})->template mutable_data<ParamT, Context>(),
        ctx());
  }

  if (sync_stats_ > 0) {
#ifdef USE_MPI
    ctx()->FinishDeviceComputation();
    if (enable_nccl_) {
#ifdef USE_NCCL
      NCCL_CHECK(ncclAllReduce(
          (void*)params,
          (void*)params,
          C_ * 2,
          this->template nccl_data_type<ParamT>(),
          ncclSum,
          this->nccl_comm(),
          ((CUDAContext*)ctx())->cuda_stream()));
#endif // USE_NCCL
    } else {
      AllReduce(params, params, C_ * 2);
    }
#endif // USE_MPI
  }

  // Gradient w.r.t. input.
  if (dX->has_name()) {
    kernels::BatchNormTrainingGrad(
        N_,
        C_,
        S_,
#ifdef USE_MPI
        float(N_ * S_ * comm_size_),
#else
        float(N_ * S_),
#endif
        data_format(),
        x,
        mu,
        rsig,
        gamma,
        params,
        params + C_,
        dy,
        dX->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void BatchNormGradientOp<Context>::RunInference() {
  using ParamT = typename math::AccumulatorType<T>::type;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto* X_params = Output("X_params")->Reshape({C_});

  auto* rv = Input(3).template data<ParamT, Context>();
  auto* params = X_params->template mutable_data<ParamT, Context>();
  ParamT *dgamma = nullptr, *dbeta = nullptr;
  if (dW->has_name() || dB->has_name()) {
    dgamma = dW->Reshape({C_})->template mutable_data<ParamT, Context>();
    dbeta = dB->Reshape({C_})->template mutable_data<ParamT, Context>();
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
      Input(2).template data<ParamT, Context>(), // mu
      params, // rsig
      Input(1).template data<ParamT, Context>(), // gamma
      Input(4).template data<T, Context>(), // dy
      dgamma,
      dbeta,
      dX->template mutable_data<T, Context>(),
      ctx());
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
