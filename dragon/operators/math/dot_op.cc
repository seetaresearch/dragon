#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void DotOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);

  if (A.ndim() == 1 && B.ndim() == 1) {
    // Compute vector product
    CHECK_EQ(A.count(), B.count())
        << "\nShapes " << A.DimString() << " and " << B.DimString()
        << " not aligned: " << A.count() << " (dim 0) != " << B.count()
        << " (dim 0)";
    math::Dot(
        A.count(),
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        Y->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  } else if (A.ndim() == 2 && B.ndim() == 2) {
    // Compute matrix multiplication
    CHECK_EQ(A.dim(1), B.dim(0))
        << "\nShapes " << A.DimString() << " and " << B.DimString()
        << " not aligned: " << A.dim(1) << " (dim 1) != " << B.dim(0)
        << " (dim 0)";
    math::Gemm(
        CblasNoTrans,
        CblasNoTrans,
        A.dim(0),
        B.dim(1),
        A.dim(1),
        1.f,
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        0.f,
        Y->Reshape({A.dim(0), B.dim(1)})->template mutable_data<T, Context>(),
        ctx());
  } else if (A.ndim() == 0 && B.ndim() == 0) {
    // Compute elementwise multiplication
    math::Mul(
        1,
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        Y->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  } else if (A.ndim() >= 2 && B.ndim() == 1) {
    // Compute matrix-vector multiplication
    CHECK_EQ(A.dim(-1), B.dim(0))
        << "\nShapes " << A.DimString() << " and " << B.DimString()
        << " not aligned: " << A.dim(-1) << " (dim -1) != " << B.dim(0)
        << " (dim 0)";
    vec64_t Y_dims(A.dims().begin(), A.dims().end() - 1);
    math::Gemv(
        CblasNoTrans,
        A.dim(0),
        A.dim(-1),
        1.f,
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        0.f,
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "\nShapes " << A.DimString() << " and " << B.DimString()
               << " not aligned.";
  }
}

template <class Context>
void DotOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void DotGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(2);
  auto *dA = Output(0), *dB = Output(1);

  if (A.ndim() == 1 && B.ndim() == 1) {
    // Gradient of vector product
    if (dA->has_name()) {
      math::Mul(
          dY.ndim(),
          dY.dims().data(),
          B.ndim(),
          B.dims().data(),
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
    if (dB->has_name()) {
      math::Mul(
          dY.ndim(),
          dY.dims().data(),
          A.ndim(),
          A.dims().data(),
          dY.template data<T, Context>(),
          A.template data<T, Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  } else if (A.ndim() == 2 && B.ndim() == 2) {
    // Gradient of matrix multiplication
    if (dA->has_name()) {
      math::Gemm(
          CblasNoTrans,
          CblasTrans,
          A.dim(0),
          A.dim(1),
          B.dim(1),
          1.f,
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          0.f,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
    if (dB->has_name()) {
      math::Gemm(
          CblasTrans,
          CblasNoTrans,
          A.dim(1),
          B.dim(1),
          A.dim(0),
          1.f,
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          0.f,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  } else if (A.ndim() == 0 && B.ndim() == 0) {
    // Gradient of elementwise multiplication
    if (dA->has_name()) {
      math::Mul(
          1,
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
    if (dB->has_name()) {
      math::Mul(
          1,
          dY.template data<T, Context>(),
          A.template data<T, Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  } else if (A.ndim() >= 2 && B.ndim() == 1) {
    // Gradient of matrix-vector multiplication
    if (dA->has_name()) {
      math::Gemm(
          CblasNoTrans,
          CblasNoTrans,
          A.dim(0),
          A.dim(1),
          1,
          1.f,
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          0.f,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
    if (dB->has_name()) {
      math::Gemv(
          CblasTrans,
          A.dim(0),
          A.dim(1),
          1.f,
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          0.f,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  } else {
    LOG(FATAL) << "\nShapes " << A.DimString() << " and " << B.DimString()
               << " not aligned.";
  }
}

template <class Context>
void DotGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(2));
}

DEPLOY_CPU_OPERATOR(Dot);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Dot);
#endif

DEPLOY_CPU_OPERATOR(DotGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DotGradient);
#endif

OPERATOR_SCHEMA(Dot)
    /* A, B */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(DotGradient)
    /* A, B, dY */
    .NumInputs(3)
    /* dA, dB */
    .NumOutputs(2);

REGISTER_GRADIENT(Dot, GenericGradientMaker);

} // namespace dragon
