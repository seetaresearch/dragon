#include "dragon/operators/math/gemm_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/filler.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void GemmOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(A);

  // Check matrix A
  auto M = transA_ ? A.count(axis) : A.count(0, axis);
  auto K = transA_ ? A.count(0, axis) : A.count(axis);

  // Check matrix B
  auto N = n_; // Init "N" from the argument
  if (N <= 0) {
    // Infer "N" from the B shape
    N = B.count() / K;
    CHECK_GT(N, 0) << "\nFailed to infer 'N' from "
                   << "the B shape: " << B.DimString();
  }
  if (transB_ > 0) {
    TENSOR_FILL(B, vec64_t({N, K}));
    CHECK(B.ndim() == 2 && B.dim(1) == K)
        << "\nMatrixB's dimensions should be [N, K].\n"
        << "Got A as (" << M << ", " << K << "), "
        << "and B as " << B.DimString();
  } else {
    TENSOR_FILL(B, vec64_t({K, N}));
    CHECK(B.ndim() == 2 && B.dim(0) == K)
        << "\nMatrixB's dimensions should be [K, N].\n"
        << "Got A as (" << M << ", " << K << "), "
        << "and B as " << B.DimString();
  }

  // Copy matrix C to Y if provided
  vec64_t Y_dims(A.dims().begin(), A.dims().begin() + axis);
  Y_dims.insert(transA_ ? Y_dims.begin() : Y_dims.end(), N);
  if (InputSize() > 2) {
    auto& C = Input(2);
    if (C.ndim() == 0) {
      TENSOR_FILL(C, vec64_t({N}));
    }
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
      LOG(FATAL) << "Could not broadcast together with shapes: "
                 << Tensor::DimString(Y_dims) << " " << C.DimString();
    }
  }

  math::Gemm(
      (CBLAS_TRANSPOSE)transA_,
      (CBLAS_TRANSPOSE)transB_,
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
void GemmOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void GemmGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(3);
  auto *dA = Output(0), *dB = Output(1), *dC = Output(2);
  CANONICALIZE_AXIS_WITH_TENSOR(A);

  // Check matrix A
  auto M = transA_ ? A.count(axis) : A.count(0, axis);
  auto K = transA_ ? A.count(0, axis) : A.count(axis);

  // Check matrix B
  auto N = n_; // Init "N" from the argument
  if (N <= 0) {
    // Infer "N" from the B shape
    N = B.count() / K;
    CHECK_GT(N, 0) << "\nFailed to infer 'N' from "
                   << "the B shape: " << B.DimString();
  }

  if (dA->has_name()) {
    if (transA_ > 0) {
      math::Gemm(
          transB_ ? CblasTrans : CblasNoTrans,
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
          transB_ ? CblasNoTrans : CblasTrans,
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
    if (transB_) {
      math::Gemm(
          CblasTrans,
          transA_ ? CblasTrans : CblasNoTrans,
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
          transA_ ? CblasNoTrans : CblasTrans,
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
    } else {
      vec32_t Y_axes, C_axes;
      math::utils::ComputeBinaryBroadcastAxes(
          dY.dims(), C.dims(), dY.dims(), Y_axes, C_axes);
      math::ReduceSum(
          dY.ndim(),
          vec32_t{dY.dims().begin(), dY.dims().end()}.data(),
          C_axes.size(),
          C_axes.data(),
          beta_,
          dY.template data<T, Context>(),
          dC->ReshapeLike(C)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
void GemmGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Gemm);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Gemm);
#endif

DEPLOY_CPU_OPERATOR(GemmGradient);
#ifdef USE_CUDA
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
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(0), I(1), I(2), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(Gemm, GradientMaker);

} // namespace dragon
