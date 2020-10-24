#include "dragon/operators/math/fully_connected_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/filler.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void FullyConnectedOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  // Determine the number of output channels
  int64_t M = X.count(0, axis), K = X.count(axis), N;
  if (out_channels_ <= 0) {
    // Infer the "N" from the weights shape
    N = W.count() / K;
    CHECK_GT(N, 0) << "\nFailed to infer the N from "
                   << "the weights shape: " << W.DimString();
  } else {
    // Use a fixed "N" from the argument
    N = out_channels_;
  }

  vec64_t Y_dims(axis + 1);
  for (int i = 0; i < axis + 1; i++) {
    Y_dims[i] = i < axis ? X.dim(i) : N;
  }

  if (transW_ > 0) {
    TENSOR_FILL(W, vec64_t({N, K}));
    CHECK(W.ndim() == 2 && W.dim(1) == K)
        << "\nWeights dimensions should be [N, K].\n"
        << "Got X as (" << M << ", " << K << "), "
        << "and W as " << W.DimString();
  } else {
    TENSOR_FILL(W, vec64_t({K, N}));
    CHECK(W.ndim() == 2 && W.dim(0) == K)
        << "\nWeights dimensions should be [K, N].\n"
        << "Got X as (" << M << ", " << K << "), "
        << "and W as " << W.DimString();
  }

  math::Gemm(
      CblasNoTrans,
      (CBLAS_TRANSPOSE)transW_,
      M,
      N,
      K,
      1.f,
      X.template data<T, Context>(),
      W.template data<T, Context>(),
      0.f,
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());

  if (InputSize() > 2) {
    TENSOR_FILL(Input(2), vec64_t({N}));
    kernel::BiasAdd(
        M,
        1,
        N,
        Y->template data<T, Context>(),
        Input(2).template data<T, Context>(),
        Y->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void FullyConnectedOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void FullyConnectedGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  // Determine the number of output channels
  int64_t M = X.count(0, axis), K = X.count(axis), N;
  if (out_channels_ <= 0) {
    // Infer the "N" from the weights shape
    N = W.count() / K;
    CHECK_GT(N, 0) << "\nFailed to infer the N from "
                   << "the weights shape: " << W.DimString();
  } else {
    // Use a fixed "N" from the argument
    N = out_channels_;
  }

  if (dX->has_name()) {
    if (transW_) {
      math::Gemm(
          CblasNoTrans,
          CblasNoTrans,
          M,
          K,
          N,
          1.f,
          dY.template data<T, Context>(),
          W.template data<T, Context>(),
          0.f,
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Gemm(
          CblasNoTrans,
          CblasTrans,
          M,
          K,
          N,
          1.f,
          dY.template data<T, Context>(),
          W.template data<T, Context>(),
          0.f,
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dW->has_name()) {
    if (transW_) {
      math::Gemm(
          CblasTrans,
          CblasNoTrans,
          N,
          K,
          M,
          1.f,
          dY.template data<T, Context>(),
          X.template data<T, Context>(),
          0.f,
          dW->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Gemm(
          CblasTrans,
          CblasNoTrans,
          K,
          N,
          M,
          1.f,
          X.template data<T, Context>(),
          dY.template data<T, Context>(),
          0.f,
          dW->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dB->has_name()) {
    vec32_t dims = {(int)M, (int)N}, axes = {0};
    math::ReduceSum(
        2,
        dims.data(),
        1,
        axes.data(),
        1.f,
        dY.template data<T, Context>(),
        dB->Reshape({N})->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void FullyConnectedGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(FullyConnected);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(FullyConnected);
#endif

DEPLOY_CPU_OPERATOR(FullyConnectedGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(FullyConnectedGradient);
#endif

OPERATOR_SCHEMA(FullyConnected)
    /* X, W, B */
    .NumInputs(2, 3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(FullyConnectedGradient)
    /* X, W, dY */
    .NumInputs(3)
    /* dX, dW, dB */
    .NumOutputs(3);

namespace {

class GradientMaker : public GradientMakerBase {
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

REGISTER_GRADIENT(FullyConnected, GradientMaker);

} // namespace dragon
