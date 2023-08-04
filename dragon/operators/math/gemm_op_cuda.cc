#ifdef USE_CUDA

#include "dragon/core/workspace.h"
#include "dragon/operators/math/gemm_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CUDAGemmOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);
  const auto A_axis = A.axis(-1), B_axis = B.axis(-1);

  // Check matrix A.
  auto M = transA_ ? A.dim(A_axis) : A.count(0, A_axis);
  auto K = transA_ ? A.count(0, A_axis) : A.dim(A_axis);

  // Check matrix B.
  auto N = transB_ ? B.count(0, B_axis) : B.dim(B_axis);
  auto K2 = transB_ ? B.dim(B_axis) : B.count(0, B_axis);
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

  // Copy matrix C to Y if not fused.
  auto fuse_bias = beta_ == 1.f && InputSize() > 2;
  if (InputSize() > 2) {
    auto& C = Input(2);
    fuse_bias = fuse_bias && C.ndim() == 1 && C.count() == N;
    if (fuse_bias) {
      Y_impl_.SetEpilogueAndBias(
          CUBLASLT_EPILOGUE_BIAS, (T*)C.template data<T, Context>());
    } else {
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
  }

  const auto beta = (InputSize() == 2 || fuse_bias) ? 0.f : beta_;
  Y_impl_.Setup<T>(transA_, transB_, alpha_, beta, A.dims(), B.dims(), ctx());
  Y_impl_.Compute<T>(
      A.template data<T, Context>(),
      B.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<Context>(Y_impl_.scratch_size()),
      ctx());
}

template <class Context>
template <typename T>
void CUDAGemmGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &C = Input(2), &dY = Input(3);
  auto *dA = Output(0), *dB = Output(1), *dC = Output(2);
  const auto A_axis = A.axis(-1), B_axis = B.axis(-1);

  // Check matrix A/B.
  auto M = transA_ ? A.dim(A_axis) : A.count(0, A_axis);
  auto K = transA_ ? A.count(0, A_axis) : A.dim(A_axis);
  auto N = transB_ ? B.count(0, B_axis) : B.dim(B_axis);

  // Check tensor C.
  auto has_bgrad = dC->has_name();
  auto fuse_bgrad = beta_ == 1.f && dC->has_name();
  fuse_bgrad = fuse_bgrad && C.ndim() == 1 && C.count() == N;

  // Compute dA.
  if (dA->has_name()) {
    if (transA_ > 0) {
      dA_impl_.Setup<T>(transB_, 1, alpha_, 0.f, B.dims(), dY.dims(), ctx());
      dA_impl_.Compute<T>(
          B.template data<T, Context>(),
          dY.template data<T, Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(dA_impl_.scratch_size()),
          ctx());
    } else {
      dA_impl_.Setup<T>(0, !transB_, alpha_, 0.f, dY.dims(), B.dims(), ctx());
      dA_impl_.Compute<T>(
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(dA_impl_.scratch_size()),
          ctx());
    }
  }

  // Compute dB and dC.
  if (dB->has_name()) {
    dB_impl_.SetEpilogueAndBias(CUBLASLT_EPILOGUE_DEFAULT, (T*)nullptr);
    if (has_bgrad && fuse_bgrad) {
#if CUBLAS_VERSION >= 11600 // CUDA >= 11.4.2
      has_bgrad = false;
      dB_impl_.SetEpilogueAndBias(
          transB_ > 0 ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BGRADA,
          dC->ReshapeLike(C)->template mutable_data<T, Context>());
#endif
    }
    if (transB_ > 0) {
      dB_impl_.Setup<T>(1, transA_, alpha_, 0.f, dY.dims(), A.dims(), ctx());
      dB_impl_.Compute<T>(
          dY.template data<T, Context>(),
          A.template data<T, Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(dB_impl_.scratch_size()),
          ctx());
    } else {
      dB_impl_.Setup<T>(!transA_, 0, alpha_, 0.f, A.dims(), dY.dims(), ctx());
      dB_impl_.Compute<T>(
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(dB_impl_.scratch_size()),
          ctx());
    }
  }

  // Compute dC.
  if (has_bgrad) {
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
          beta_,
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

INSTANTIATE_OPERATOR(CUDAGemm, CUDAContext);
INSTANTIATE_OPERATOR(CUDAGemmGradient, CUDAContext);
REGISTER_CUDA_OPERATOR(Gemm, CUDAGemmOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(GemmGradient, CUDAGemmGradientOp<CUDAContext>);

} // namespace dragon

#endif // USE_CUDA
