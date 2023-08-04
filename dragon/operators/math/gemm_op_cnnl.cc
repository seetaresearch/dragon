#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/math/gemm_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLGemmOp<Context>::DoRunWithType() {
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
    CHECK(math::utils::IsBinaryBroadcast(Y_dims, C.dims(), Y_dims))
        << "\nCould not broadcast together with shapes: "
        << Tensor::DimString(Y_dims) << " " << C.DimString();
    math::Set(
        C.ndim(),
        C.dims().data(),
        Y_dims.size(),
        Y_dims.data(),
        C.template data<T, Context>(),
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  }

  impl_.Setup<T>(
      transA_ > 0 ? CblasTrans : CblasNoTrans,
      transB_ > 0 ? CblasTrans : CblasNoTrans,
      alpha_,
      InputSize() > 2 ? beta_ : 0.f,
      transA_ ? vec64_t({K, M}) : vec64_t({M, K}),
      transB_ ? vec64_t({N, K}) : vec64_t({K, N}),
      ctx());
  impl_.Compute<T>(
      A.template data<T, Context>(),
      B.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<Context>(impl_.scratch_size()),
      ctx());
}

template <class Context>
template <typename T>
void CNNLGemmGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(3);
  auto *dA = Output(0), *dB = Output(1), *dC = Output(2);
  const auto A_axis = A.axis(-1), B_axis = B.axis(-1);

  // Check matrix A/B.
  auto M = transA_ ? A.count(A_axis) : A.count(0, A_axis);
  auto K = transA_ ? A.count(0, A_axis) : A.count(A_axis);
  auto N = transB_ ? B.count(0, B_axis) : B.count(B_axis);

  if (dA->has_name()) {
    if (transA_ > 0) {
      mm_impl_.Setup<T>(
          transB_ > 0 ? CblasTrans : CblasNoTrans,
          CblasTrans,
          alpha_,
          0.f,
          transB_ > 0 ? vec64_t({N, K}) : vec64_t({K, N}),
          vec64_t({M, N}),
          ctx());
      mm_impl_.Compute<T>(
          B.template data<T, Context>(),
          dY.template data<T, Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
          ctx());
    } else {
      mm_impl_.Setup<T>(
          CblasNoTrans,
          transB_ > 0 ? CblasNoTrans : CblasTrans,
          alpha_,
          0.f,
          vec64_t({M, N}),
          transB_ > 0 ? vec64_t({N, K}) : vec64_t({K, N}),
          ctx());
      mm_impl_.Compute<T>(
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (transB_ > 0) {
      mm_impl_.Setup<T>(
          CblasTrans,
          transA_ > 0 ? CblasTrans : CblasNoTrans,
          alpha_,
          0.f,
          vec64_t({M, N}),
          transA_ > 0 ? vec64_t({K, M}) : vec64_t({M, K}),
          ctx());
      mm_impl_.Compute<T>(
          dY.template data<T, Context>(),
          A.template data<T, Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
          ctx());
    } else {
      mm_impl_.Setup<T>(
          transA_ > 0 ? CblasNoTrans : CblasTrans,
          CblasNoTrans,
          alpha_,
          0.f,
          transA_ > 0 ? vec64_t({K, M}) : vec64_t({M, K}),
          vec64_t({M, N}),
          ctx());
      mm_impl_.Compute<T>(
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
          ctx());
    }
  }

  if (dC->has_name()) {
    auto& C = Input(2);
    vec64_t Y_axes, C_axes;
    math::utils::ComputeBroadcastAxes(
        dY.dims(), C.dims(), dY.dims(), Y_axes, C_axes);
    if (C_axes.empty()) {
      math::Scale(
          C.count(),
          beta_,
          dY.template data<T, Context>(),
          dC->ReshapeLike(C)->template mutable_data<T, Context>(),
          ctx());
    } else {
      reduce_impl_.Setup<T>(dY.dims(), C_axes, ctx());
      reduce_impl_.Compute<T>(
          dY.template data<T, Context>(),
          dC->ReshapeLike(C)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(
              reduce_impl_.scratch_size()),
          ctx(),
          beta_);
    }
  }
}

DEPLOY_CNNL_OPERATOR(Gemm);
DEPLOY_CNNL_OPERATOR(GemmGradient);

} // namespace dragon

#endif // USE_MLU
