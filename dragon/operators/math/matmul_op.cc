#include "dragon/operators/math/matmul_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MatMulOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);

  CHECK_GE(A.ndim(), 2) << "\nTensor(" << A.name() + ") must be a matrix"
                        << "(or rank > 2, representing batches of matrices).";
  CHECK_GE(B.ndim(), 2) << "\nTensor(" << B.name() + ") must be a matrix"
                        << "(or rank > 2, representing batches of matrices).";

  auto M1 = A.dim(-2), N1 = A.dim(-1);
  auto M2 = B.dim(-2), N2 = B.dim(-1);
  auto M = transA_ ? N1 : M1, N = transB_ ? M2 : N2;
  auto K1 = transA_ ? M1 : N1, K2 = transB_ ? N2 : M2;
  auto A_stride = M1 * N1, B_stride = M2 * N2, Y_stride = M * N;
  auto batch_size = A.count() / A_stride;

  CHECK((K1 == K2) && (batch_size == (B.count() / B_stride)))
      << "\nTensor(" << A.name() << "): " << A.DimString()
      << " can not mul with Tensor"
      << "(" << B.name() << "): " << B.DimString();

  vec64_t Y_dims(A.dims());
  Y_dims[Y_dims.size() - 2] = M;
  Y_dims[Y_dims.size() - 1] = N;
  Y->Reshape(Y_dims);

  auto* a = A.template data<T, Context>();
  auto* b = B.template data<T, Context>();
  auto* y = Y->template mutable_data<T, Context>();

  for (int i = 0; i < batch_size; ++i) {
    math::Gemm(
        transA_ > 0 ? CblasTrans : CblasNoTrans,
        transB_ > 0 ? CblasTrans : CblasNoTrans,
        M,
        N,
        K1,
        1.f,
        a + i * A_stride,
        b + i * B_stride,
        0.f,
        y + i * Y_stride,
        ctx());
  }
}

template <class Context>
void MatMulOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void MatMulGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(2);
  auto *dA = Output(0), *dB = Output(1);

  CHECK_GE(A.ndim(), 2) << "\nTensor(" << A.name() + ") must be a matrix"
                        << "(or rank > 2, representing batches of matrices).";
  CHECK_GE(B.ndim(), 2) << "\nTensor(" << B.name() + ") must be a matrix"
                        << "(or rank > 2, representing batches of matrices).";

  auto M1 = A.dim(-2), N1 = A.dim(-1);
  auto M2 = B.dim(-2), N2 = B.dim(-1);
  auto M = transA_ ? N1 : M1, N = transB_ ? M2 : N2;
  auto K1 = transA_ ? M1 : N1, K2 = transB_ ? N2 : M2;
  auto A_stride = M1 * N1, B_stride = M2 * N2, Y_stride = M * N;
  auto batch_size = A.count() / A_stride;

  CHECK((K1 == K2) && (batch_size == (B.count() / B_stride)))
      << "\nTensor(" << A.name() << "): " << A.DimString()
      << " can not mul with Tensor"
      << "(" << B.name() << "): " << B.DimString();

  if (dA->has_name()) {
    auto* b = B.template data<T, Context>();
    auto* dy = dY.template data<T, Context>();
    auto* da = dA->ReshapeLike(A)->template mutable_data<T, Context>();
    if (transA_ > 0) {
      for (int i = 0; i < batch_size; ++i) {
        math::Gemm(
            transB_ ? CblasTrans : CblasNoTrans,
            CblasTrans,
            K1,
            M,
            N,
            1.f,
            b + i * B_stride,
            dy + i * Y_stride,
            0.f,
            da + i * A_stride,
            ctx());
      }
    } else {
      for (int i = 0; i < batch_size; ++i) {
        math::Gemm(
            CblasNoTrans,
            transB_ ? CblasNoTrans : CblasTrans,
            M,
            K1,
            N,
            1.f,
            dy + i * Y_stride,
            b + i * B_stride,
            0.f,
            da + i * A_stride,
            ctx());
      }
    }

    if (dB->has_name()) {
      auto* a = A.template data<T, Context>();
      auto* dy = dY.template data<T, Context>();
      auto* db = dB->ReshapeLike(B)->template mutable_data<T, Context>();
      if (transB_) {
        for (int i = 0; i < batch_size; ++i) {
          math::Gemm(
              CblasTrans,
              transA_ ? CblasTrans : CblasNoTrans,
              N,
              K1,
              M,
              1.f,
              dy + i * Y_stride,
              a + i * A_stride,
              0.f,
              db + i * B_stride,
              ctx());
        }
      } else {
        for (int i = 0; i < batch_size; ++i) {
          math::Gemm(
              transA_ ? CblasNoTrans : CblasTrans,
              CblasNoTrans,
              K1,
              N,
              M,
              1.f,
              a + i * A_stride,
              dy + i * Y_stride,
              0.f,
              db + i * B_stride,
              ctx());
        }
      }
    }
  }
}

template <class Context>
void MatMulGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(MatMul);
#ifdef USE_CUDA
DEPLOY_CUDA(MatMul);
#endif

DEPLOY_CPU(MatMulGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(MatMulGradient);
#endif

OPERATOR_SCHEMA(MatMul)
    /* A, B */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(MatMulGradient)
    /* A, B, dY */
    .NumInputs(3)
    /* dA, dB */
    .NumOutputs(2);

REGISTER_GRADIENT(MatMul, GenericGradientMaker);

} // namespace dragon
