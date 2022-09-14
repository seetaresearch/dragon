#include "dragon/operators/math/gemm_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void GemmOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);
  const auto A_axis = A.axis(-1), B_axis = B.axis(-1);

  // Check matrix A.
  auto M = transA_ ? A.count(A_axis) : A.count(0, A_axis);
  auto K = transA_ ? A.count(0, A_axis) : A.count(A_axis);

  // Check matrix B.
  auto N = transB_ ? B.count(0, B_axis) : B.count(B_axis);
  auto K2 = transB_ ? B.count(B_axis) : B.count(0, B_axis);
  if (transB_) {
    CHECK_EQ(K, K2) << "\nMatrixB's dimensions should be (...," << K
                    << "), got " << B.DimString() << ".";
  } else {
    CHECK_EQ(K, K2) << "\nMatrixB's dimensions should be reshaped to (" << K
                    << "," << N << "), got " << B.DimString() << ".";
  }

  vec64_t Y_dims;
  if (transA_) {
    Y_dims.push_back(M);
  } else {
    Y_dims.insert(Y_dims.end(), A.dims().begin(), A.dims().begin() + A_axis);
  }
  if (transB_) {
    Y_dims.insert(Y_dims.end(), B.dims().begin(), B.dims().begin() + B_axis);
  } else {
    Y_dims.push_back(N);
  }

  // Copy matrix C to Y if provided.
  if (InputSize() > 2) {
    auto& C = Input(2);
    if (math::utils::IsBinaryBroadcast(Y_dims, C.dims(), Y_dims)) {
      math::Set(
          C.ndim(),
          C.dims().data(),
          Y_dims.size(),
          Y_dims.data(),
          C.template data<T, Context>(),
          Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "Could not broadcast with shapes: "
                 << Tensor::DimString(Y_dims) << " " << C.DimString();
    }
  }

  math::Gemm(
      transA_ > 0 ? CblasTrans : CblasNoTrans,
      transB_ > 0 ? CblasTrans : CblasNoTrans,
      M,
      N,
      K,
      alpha_,
      A.template data<T, Context>(),
      B.template data<T, Context>(),
      InputSize() > 2 ? beta_ : 0.f,
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void GemmGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(3);
  auto *dA = Output(0), *dB = Output(1), *dC = Output(2);
  const auto A_axis = A.axis(-1), B_axis = B.axis(-1);

  // Check matrix A/B.
  auto M = transA_ ? A.count(A_axis) : A.count(0, A_axis);
  auto K = transA_ ? A.count(0, A_axis) : A.count(A_axis);
  auto N = transB_ ? B.count(0, B_axis) : B.count(B_axis);

  if (dA->has_name()) {
    if (transA_ > 0) {
      math::Gemm(
          transB_ > 0 ? CblasTrans : CblasNoTrans,
          CblasTrans,
          K,
          M,
          N,
          alpha_,
          B.template data<T, Context>(),
          dY.template data<T, Context>(),
          0.f,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Gemm(
          CblasNoTrans,
          transB_ > 0 ? CblasNoTrans : CblasTrans,
          M,
          K,
          N,
          alpha_,
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          0.f,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (transB_ > 0) {
      math::Gemm(
          CblasTrans,
          transA_ > 0 ? CblasTrans : CblasNoTrans,
          N,
          K,
          M,
          alpha_,
          dY.template data<T, Context>(),
          A.template data<T, Context>(),
          0.f,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Gemm(
          transA_ > 0 ? CblasNoTrans : CblasTrans,
          CblasNoTrans,
          K,
          N,
          M,
          alpha_,
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          0.f,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dC->has_name()) {
    auto& C = Input(2);
    if (C.count() == dY.count()) {
      math::Scale(
          dY.count(),
          beta_,
          dY.template data<T, Context>(),
          dC->ReshapeLike(C)->template mutable_data<T, Context>(),
          ctx());
    } else if (C.ndim() == 1 && C.count() == N) {
      auto* Z = ctx()->workspace()->CreateTensor("Ones");
      if (Z->count() < M) {
        math::Set(
            M,
            convert::To<T>(1.f),
            Z->Reshape({M})->template mutable_data<T, Context>(),
            ctx());
      }
      math::Gemv(
          CblasTrans,
          M,
          N,
          1.f,
          dY.template data<T, Context>(),
          Z->template mutable_data<T, Context>(),
          0.f,
          dC->ReshapeLike(C)->template mutable_data<T, Context>(),
          ctx());
    } else {
      vec32_t Y_axes, C_axes;
      math::utils::ComputeBroadcastAxes(
          dY.dims(), C.dims(), dY.dims(), Y_axes, C_axes);
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          C_axes.size(),
          vec64_t({C_axes.begin(), C_axes.end()}).data(),
          beta_,
          dY.template data<T, Context>(),
          dC->ReshapeLike(C)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

DEPLOY_CPU_OPERATOR(Gemm);
DEPLOY_CPU_OPERATOR(GemmGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Gemm);
DEPLOY_CUDA_OPERATOR(GemmGradient);
#endif

OPERATOR_SCHEMA(Gemm)
    /* A, B, C */
    .NumInputs(2, 3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(GemmGradient)
    /* A, B, C, dY */
    .NumInputs(4)
    /* dA, dB, dC */
    .NumOutputs(3);

namespace {

class GradientMaker : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), I(2), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(Gemm, GradientMaker);

} // namespace dragon
