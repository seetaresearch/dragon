#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/math/matmul_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLMatMulOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);
  auto A_ndim = A.ndim(), B_ndim = B.ndim();

  if (A_ndim == 1 || B_ndim == 1) {
    LOG(FATAL) << "Matrix-Vector multiplication is not supported.";
  }

  // Check matrix A.
  const auto M = A.dim(A_ndim - 2);
  const auto K = A.dim(A_ndim - 1);

  // Check matrix B.
  CHECK_EQ(B.dim(B_ndim - 2), K) << "\nExcept the second last dim of B is " << K
                                 << ", got " << B.dim(B_ndim - 2);
  const auto N = B.dim(B_ndim - 1);

  // Check batching && broadcasting.
  vec64_t A_batch_dims(A.dims().begin(), A.dims().end() - 2);
  vec64_t B_batch_dims(B.dims().begin(), B.dims().end() - 2), Y_dims;
  CHECK(math::utils::IsBinaryBroadcast(A_batch_dims, B_batch_dims, Y_dims))
      << "\nCould not broadcast with " << A.DimString() << " " << B.DimString();
  const auto A_batch_size = math::utils::Prod(A_batch_dims);
  const auto B_batch_size = math::utils::Prod(B_batch_dims);
  Y_dims.push_back(M);
  Y_dims.push_back(N);

  if (B_batch_size == 1) {
    // Batched Matrix @ Broadcasted Matrix.
    mm_impl_.Setup<T>(
        CblasNoTrans,
        CblasNoTrans,
        1.f,
        0.f,
        vec64_t({A_batch_size * M, K}),
        vec64_t({K, N}),
        ctx());
    mm_impl_.Compute<T>(
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
        ctx());
  } else {
    // Batched Matrix @ Batched Matrix.
    bmm_impl_.Setup<T>(0, 0, 1.f, 0.f, A.dims(), B.dims(), ctx());
    bmm_impl_.Compute<T>(
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
        ctx());
  }
}

template <class Context>
template <typename T>
void CNNLMatMulGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(2);
  auto *dA = Output(0), *dB = Output(1);
  auto A_ndim = A.ndim(), B_ndim = B.ndim();

  if (A_ndim == 1 || B_ndim == 1) {
    LOG(FATAL) << "Matrix-Vector multiplication is not supported.";
  }

  // Check matrix A && B.
  const auto M = A.dim(A_ndim - 2);
  const auto K = A.dim(A_ndim - 1);
  const auto N = B.dim(B_ndim - 1);

  // Check batching && broadcasting.
  vec64_t A_bcast_axes, B_bcast_axes;
  vec64_t A_bcast_dims, B_bcast_dims, Y_dims;
  vec64_t A_batch_dims(A.dims().begin(), A.dims().end() - 2);
  vec64_t B_batch_dims(B.dims().begin(), B.dims().end() - 2);
  CHECK(math::utils::IsBinaryBroadcast(A_batch_dims, B_batch_dims, Y_dims))
      << "\nCould not broadcast with " << A.DimString() << " " << B.DimString();
  math::utils::ComputeBroadcastDims(
      A_batch_dims, B_batch_dims, A_bcast_dims, B_bcast_dims);
  math::utils::ComputeBroadcastAxes(
      A_bcast_dims, B_bcast_dims, Y_dims, A_bcast_axes, B_bcast_axes);
  const auto A_batch_size = math::utils::Prod(A_bcast_dims);
  const auto B_batch_size = math::utils::Prod(B_bcast_dims);
  const auto Y_batch_size = math::utils::Prod(Y_dims);

  if (B_batch_size == 1) {
    // Batched Matrix @ Broadcasted Matrix.
    if (dA->has_name()) {
      mm_impl_.Setup<T>(
          CblasNoTrans,
          CblasTrans,
          1.f,
          0.f,
          vec64_t({A_batch_size * M, N}),
          vec64_t({K, N}),
          ctx());
      mm_impl_.Compute<T>(
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
          ctx());
    }
    if (dB->has_name()) {
      mm_impl_.Setup<T>(
          CblasTrans,
          CblasNoTrans,
          1.f,
          0.f,
          vec64_t({A_batch_size * M, K}),
          vec64_t({A_batch_size * M, N}),
          ctx());
      mm_impl_.Compute<T>(
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
          ctx());
    }
    return;
  }

  int64_t buffer_count = 0;
  T *data = nullptr, *buffer = nullptr;
  if (dA->has_name() && A_batch_size != Y_batch_size) {
    buffer_count = Y_batch_size * M * K;
  }
  if (dB->has_name() && B_batch_size != Y_batch_size) {
    buffer_count = std::max(buffer_count, Y_batch_size * K * N);
  }
  buffer = ctx()->workspace()->template data<T, Context>(
      {buffer_count}, "BufferKernel");

  if (dA->has_name()) {
    data = dA->ReshapeLike(A)->template mutable_data<T, Context>();
    bmm_impl_.Setup<T>(0, 1, 1.f, 0.f, dY.dims(), B.dims(), ctx());
    bmm_impl_.Compute<T>(
        dY.template data<T, Context>(),
        B.template data<T, Context>(),
        A_batch_size != Y_batch_size ? buffer : data,
        ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
        ctx());
  }

  if (dA->has_name() && A_batch_size != Y_batch_size) {
    A_bcast_dims = Y_dims;
    A_bcast_dims.push_back(M);
    A_bcast_dims.push_back(K);
    reduce_impl_.Setup<T>(CNNL_REDUCE_ADD, A_bcast_dims, A_bcast_axes, ctx());
    reduce_impl_.Compute<T>(
        buffer,
        data,
        ctx()->workspace()->template data<Context>(reduce_impl_.scratch_size()),
        ctx());
  }

  if (dB->has_name()) {
    data = dB->ReshapeLike(B)->template mutable_data<T, Context>();
    bmm_impl_.Setup<T>(1, 0, 1.f, 0.f, A.dims(), dY.dims(), ctx());
    bmm_impl_.Compute<T>(
        A.template data<T, Context>(),
        dY.template data<T, Context>(),
        B_batch_size != Y_batch_size ? buffer : data,
        ctx()->workspace()->template data<Context>(mm_impl_.scratch_size()),
        ctx());
  }

  if (dB->has_name() && B_batch_size != Y_batch_size) {
    B_bcast_dims = Y_dims;
    B_bcast_dims.push_back(K);
    B_bcast_dims.push_back(N);
    reduce_impl_.Setup<T>(CNNL_REDUCE_ADD, B_bcast_dims, B_bcast_axes, ctx());
    reduce_impl_.Compute<T>(
        buffer,
        data,
        ctx()->workspace()->template data<Context>(reduce_impl_.scratch_size()),
        ctx());
  }
}

DEPLOY_CNNL_OPERATOR(MatMul);
DEPLOY_CNNL_OPERATOR(MatMulGradient);

} // namespace dragon

#endif // USE_MLU
