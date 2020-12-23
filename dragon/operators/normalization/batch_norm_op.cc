#include "dragon/operators/normalization/batch_norm_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/filler.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void BatchNormOp<Context>::TrainingImpl() {
  using ParamT = typename math::utils::AccmulatorType<T>::type;
  TENSOR_FILL_WITH_TYPE(Input(1), vec64_t({C_}), ParamT);
  TENSOR_FILL_WITH_TYPE(Input(2), vec64_t({C_}), ParamT);
  TENSOR_FILL_WITH_TYPE(Input(3), vec64_t({C_}), ParamT);
  TENSOR_FILL_WITH_TYPE(Input(4), vec64_t({C_}), ParamT);

  auto* X_mu = Buffer("X_mu")->Reshape({C_});
  auto* X_rsig = Buffer("X_rsig")->Reshape({C_});
  auto* X_scale = Buffer("X_scale")->Reshape({C_});
  auto* X_bias = Buffer("X_bias")->Reshape({C_});

  auto* x = Input(0).template data<T, Context>();
  auto* rm = Input(3).template mutable_data<ParamT, Context>();
  auto* rv = Input(4).template mutable_data<ParamT, Context>();
  auto* mu = X_mu->template mutable_data<ParamT, Context>();
  auto* rsig = X_rsig->template mutable_data<ParamT, Context>();
  auto* scale = X_scale->template mutable_data<ParamT, Context>();

  // Compute moments
  if (sync_stats_ > 0) {
#ifdef USE_MPI
    // Compute E(X) and E(X^2)
    kernel::BatchNormExpectation(
        N_,
        C_,
        S_,
        float(N_ * S_ * comm_size_),
        data_format(),
        x,
        mu,
        rsig,
        ctx());
    // Compute D(X) = E(X^2) - E(X)^2
    ctx()->FinishDeviceComputation();
    if (enable_nccl_) {
#ifdef USE_NCCL
      auto coll_comm = this->nccl_comm();
      auto coll_dtype = this->template nccl_dtype<ParamT>();
      NCCL_CHECK(ncclAllReduce(
          (void*)mu,
          (void*)mu,
          C_,
          coll_dtype,
          ncclSum,
          coll_comm,
          ((CUDAContext*)ctx())->cuda_stream()));
      NCCL_CHECK(ncclAllReduce(
          (void*)rsig,
          (void*)rsig,
          C_,
          coll_dtype,
          ncclSum,
          coll_comm,
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
    auto decay_factor = momentum();
    math::Axpby(C_, 1.f - decay_factor, mu, decay_factor, rm, ctx());
    math::Axpby(C_, 1.f - decay_factor, rsig, decay_factor, rv, ctx());
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
      Input(1).template data<ParamT, Context>(), // gamma
      Input(2).template data<ParamT, Context>(), // beta
      scale,
      X_bias->template mutable_data<ParamT, Context>(),
      Output(0)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void BatchNormOp<Context>::InferenceImpl() {
  using ParamT = typename math::utils::AccmulatorType<T>::type;
  TENSOR_FILL_WITH_TYPE(Input(1), vec64_t({C_}), ParamT);
  TENSOR_FILL_WITH_TYPE(Input(2), vec64_t({C_}), ParamT);
  TENSOR_FILL_WITH_TYPE(Input(3), vec64_t({C_}), ParamT);
  TENSOR_FILL_WITH_TYPE(Input(4), vec64_t({C_}), ParamT);

  auto* X_rsig = Buffer("X_rsig")->Reshape({C_});
  auto* X_scale = Buffer("X_scale")->Reshape({C_});
  auto* X_bias = Buffer("X_bias")->Reshape({C_});
  auto* rv = Input(4).template data<ParamT, Context>();
  auto* rsig = X_rsig->template mutable_data<ParamT, Context>();

  // Inverse stddev from variance
  math::InvStd(C_, epsilon_, rv, rsig, ctx());

  // Fuse parameters to compute affine transformation
  kernel::BatchNorm(
      N_,
      C_,
      S_,
      data_format(),
      Input(0).template data<T, Context>(),
      Input(3).template data<ParamT, Context>(),
      rsig,
      Input(1).template data<ParamT, Context>(), // gamma
      Input(2).template data<ParamT, Context>(), // beta
      X_scale->template mutable_data<ParamT, Context>(),
      X_bias->template mutable_data<ParamT, Context>(),
      Output(0)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void BatchNormOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Get the recomputing flag
  auto* flag = workspace()->GetTensor("/share/flag/recomputing");
  is_recomputing_ = flag->template data<bool, CPUContext>()[0] ? 1 : 0;

  // Dispatch the training or inference implementation
  Output(0)->ReshapeLike(Input(0));
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void BatchNormGradientOp<Context>::TrainingImpl() {
  using ParamT = typename math::utils::AccmulatorType<T>::type;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto *X_mu = Buffer("X_mu"), *X_rsig = Buffer("X_rsig");

  auto* x = Input(0).template data<T, Context>();
  auto* gamma = Input(1).template data<ParamT, Context>();
  auto* dy = Input(4).template data<T, Context>();
  auto* mu = X_mu->template data<ParamT, Context>();
  auto* rsig = X_rsig->template data<ParamT, Context>();
  auto* dgamma = dW->Reshape({C_})->template mutable_data<ParamT, Context>();
  auto* dbeta = dB->Reshape({C_})->template mutable_data<ParamT, Context>();

  // Gradient w.r.t. gamma and beta
  kernel::BatchNormWGrad(
      N_, C_, S_, data_format(), x, mu, rsig, dy, dgamma, dbeta, ctx());

  if (sync_stats_ > 0) {
#ifdef USE_MPI
    ctx()->FinishDeviceComputation();
    if (enable_nccl_) {
#ifdef USE_NCCL
      auto coll_comm = this->nccl_comm();
      auto coll_dtype = this->template nccl_dtype<ParamT>();
      NCCL_CHECK(ncclAllReduce(
          (void*)dgamma,
          (void*)dgamma,
          C_,
          coll_dtype,
          ncclSum,
          coll_comm,
          ((CUDAContext*)ctx())->cuda_stream()));
      NCCL_CHECK(ncclAllReduce(
          (void*)dbeta,
          (void*)dbeta,
          C_,
          coll_dtype,
          ncclSum,
          coll_comm,
          ((CUDAContext*)ctx())->cuda_stream()));
#endif // USE_NCCL
    } else {
      AllReduce(dgamma, dgamma, C_);
      AllReduce(dbeta, dbeta, C_);
    }
#endif // USE_MPI
  }

  // Gradient w.r.t. input
  kernel::BatchNormTrainingGrad(
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
      dgamma,
      dbeta,
      dy,
      Output(0)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void BatchNormGradientOp<Context>::InferenceImpl() {
  using ParamT = typename math::utils::AccmulatorType<T>::type;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto* X_scale = Buffer("X_scale")->Reshape({C_});

  auto* rv = Input(3).template data<ParamT, Context>();
  auto* rsig = X_scale->template mutable_data<ParamT, Context>();

  // Gradient w.r.t. gamma or beta if necessary
  ParamT *dgamma = nullptr, *dbeta = nullptr;
  if (dW->has_name() || dB->has_name()) {
    dgamma = dW->Reshape({C_})->template mutable_data<ParamT, Context>();
    dbeta = dB->Reshape({C_})->template mutable_data<ParamT, Context>();
  }

  // Inverse stddev from variance
  math::InvStd(C_, epsilon_, rv, rsig, ctx());

  // Gradient w.r.t. gamma, beta and input
  kernel::BatchNormInferenceGrad(
      N_,
      C_,
      S_,
      data_format(),
      Input(0).template data<T, Context>(), // x
      Input(2).template data<ParamT, Context>(), // rm
      rsig,
      Input(1).template data<ParamT, Context>(), // gamma
      Input(4).template data<T, Context>(), // dy
      dgamma,
      dbeta,
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void BatchNormGradientOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Dispatch the training or inference implementation
  Output(0)->ReshapeLike(Input(0));
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
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
