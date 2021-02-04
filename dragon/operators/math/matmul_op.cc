#include "dragon/operators/math/matmul_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MatMulOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);
  auto A_ndim = A.ndim(), B_ndim = B.ndim();

  if (A_ndim == 1 && B_ndim == 1) {
    // Vector x Vector
    CHECK_EQ(A.count(), B.count()) << "\nExcept equal length of two vectors.";
    math::Dot(
        A.count(),
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        Y->Reshape({})->template mutable_data<T, Context>(),
        ctx());
    return;
  }

  if (A_ndim == 1) {
    const auto N = A.count();
    CHECK_EQ(B.dim(B_ndim - 2), N) << "\nExcept the second last dim of B is "
                                   << N << ", got " << B.dim(B_ndim - 2);
    const auto M = B.dim(B_ndim - 1);
    const auto batch_size = B.count() / (M * N);
    vec64_t Y_dims(B.dims().begin(), B.dims().end() - 1);
    Y_dims.back() = B.dims().back();
    if (batch_size == 1) {
      // Vector x Matrix
      math::Gemv(
          CblasTrans,
          N,
          M,
          1.f,
          B.template data<T, Context>(),
          A.template data<T, Context>(),
          0.f,
          Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
          ctx());
    } else {
      // Broadcasted Vector x Batched Matrix
      math::GemmStridedBatched(
          CblasTrans,
          CblasNoTrans,
          batch_size,
          M,
          1,
          N,
          M * N,
          0,
          M,
          1.f,
          B.template data<T, Context>(),
          A.template data<T, Context>(),
          0.f,
          Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
          ctx());
    }
    return;
  }

  if (B_ndim == 1) {
    // Matrix x Vector
    const auto N = B.count();
    CHECK_EQ(A.dim(A_ndim - 1), N) << "\nExcept the last dim of A is " << N
                                   << ", got " << A.dim(A_ndim - 1);
    const auto M = A.count() / N;
    vec64_t Y_dims(A.dims());
    Y_dims.erase(Y_dims.end() - 1);
    math::Gemv(
        CblasNoTrans,
        M,
        N,
        1.f,
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        0.f,
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
    return;
  }

  // Check matrix A
  const auto M = A.dim(A_ndim - 2);
  const auto K = A.dim(A_ndim - 1);

  // Check matrix B
  CHECK_EQ(B.dim(B_ndim - 2), K) << "\nExcept the second last dim of B is " << K
                                 << ", got " << B.dim(B_ndim - 2);
  const auto N = B.dim(B_ndim - 1);

  // Check batching && broadcasting
  vec64_t A_dims(A.dims().begin(), A.dims().end() - 2);
  vec64_t B_dims(B.dims().begin(), B.dims().end() - 2);
  vec64_t A_batch_dims, B_batch_dims, Y_dims;
  if (math::utils::IsBinaryBroadcast(A_dims, B_dims, Y_dims)) {
    math::utils::ComputeBinaryBroadcastDims(
        A_dims, B_dims, A_batch_dims, B_batch_dims);
  } else {
    LOG(FATAL) << "Could not broadcast together with shapes " << A.DimString()
               << " " << B.DimString();
  }
  const int64_t batch_ndim = A_batch_dims.size();
  const bool broadcasting = A_batch_dims != B_batch_dims;
  Y_dims.push_back(M);
  Y_dims.push_back(N);

  const auto A_batch_size = std::accumulate(
      A_batch_dims.begin(),
      A_batch_dims.end(),
      1LL,
      std::multiplies<int64_t>());
  const auto B_batch_size = std::accumulate(
      B_batch_dims.begin(),
      B_batch_dims.end(),
      1LL,
      std::multiplies<int64_t>());
  const auto Y_batch_size = std::accumulate(
      Y_dims.begin(),
      Y_dims.begin() + batch_ndim,
      1LL,
      std::multiplies<int64_t>());

  if (B_batch_size == 1) {
    // Batched Matrix x Broadcasted Matrix
    math::Gemm(
        CblasNoTrans,
        CblasNoTrans,
        A_batch_size * M,
        N,
        K,
        1.f,
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        0.f,
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else if (A_batch_size == 1) {
    // Broadcasted Matrix x Batched Matrix
    math::GemmStridedBatched(
        CblasNoTrans,
        CblasNoTrans,
        Y_batch_size,
        M,
        N,
        K,
        0,
        K * N,
        M * N,
        1.f,
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        0.0f,
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else if (!broadcasting) {
    // Batched Matrix x Batched Matrix
    math::GemmStridedBatched(
        CblasNoTrans,
        CblasNoTrans,
        Y_batch_size,
        M,
        N,
        K,
        M * K,
        K * N,
        M * N,
        1.f,
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        0.f,
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else {
    // Broadcasted Matrix x Broadcasted Matrix
    vector<const T*> A_arr(Y_batch_size);
    vector<const T*> B_arr(Y_batch_size);
    vector<T*> Y_arr(Y_batch_size);
    vec64_t index(batch_ndim, 0);
    auto* A_data = A.template data<T, Context>();
    auto* B_data = B.template data<T, Context>();
    auto* Y_data = Y->Reshape(Y_dims)->template mutable_data<T, Context>();
    for (int64_t Y_i = 0; Y_i < Y_batch_size; ++Y_i) {
      const auto A_i = math::utils::GetIndexFromDims(
          batch_ndim, A_batch_dims.data(), index.data());
      const auto B_i = math::utils::GetIndexFromDims(
          batch_ndim, B_batch_dims.data(), index.data());
      A_arr[Y_i] = A_data + A_i * M * K;
      B_arr[Y_i] = B_data + B_i * K * N;
      Y_arr[Y_i] = Y_data + Y_i * M * N;
      math::utils::IncreaseIndexInDims(batch_ndim, Y_dims.data(), index.data());
    }
    math::GemmBatched(
        CblasNoTrans,
        CblasNoTrans,
        Y_batch_size,
        M,
        N,
        K,
        1.f,
        A_arr.data(),
        B_arr.data(),
        0.f,
        Y_arr.data(),
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
  auto A_ndim = A.ndim(), B_ndim = B.ndim();

  if (A_ndim == 1 && B_ndim == 1) {
    // Vector x Vector
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
    return;
  }

  if (A_ndim == 1) {
    const auto N = A.count();
    if (dA->has_name()) {
      const auto M = B.dim(B_ndim - 1);
      const auto batch_size = B.count() / (M * N);
      if (batch_size == 1) {
        // Vector x Matrix
        math::Gemv(
            CblasNoTrans,
            N,
            M,
            1.f,
            B.template data<T, Context>(),
            dY.template data<T, Context>(),
            0.f,
            dA->ReshapeLike(A)->template mutable_data<T, Context>(),
            ctx());
      } else {
        // Broadcasted Vector x Batched Matrix
        auto* scratch =
            ctx()->workspace()->template data<T, Context>({batch_size * N})[0];
        math::GemmStridedBatched(
            CblasNoTrans,
            CblasNoTrans,
            batch_size,
            N,
            1,
            M,
            M * N,
            M,
            N,
            1.f,
            B.template data<T, Context>(),
            dY.template data<T, Context>(),
            0.f,
            scratch,
            ctx());
        math::ReduceSum(
            2,
            vec32_t{int(batch_size), int(N)}.data(),
            1,
            vec32_t{0}.data(),
            1.f,
            scratch,
            dA->ReshapeLike(A)->template mutable_data<T, Context>(),
            ctx());
      }
    }
    if (dB->has_name()) {
      const auto M = B.dim(B_ndim - 1);
      const auto batch_size = B.count() / (M * N);
      if (batch_size == 1) {
        // Vector x Matrix
        math::Gemm(
            CblasNoTrans,
            CblasNoTrans,
            N,
            M,
            1,
            1.f,
            A.template data<T, Context>(),
            dY.template data<T, Context>(),
            0.f,
            dB->ReshapeLike(B)->template mutable_data<T, Context>(),
            ctx());
      } else {
        // Broadcasted Vector x Batched Matrix
        math::GemmStridedBatched(
            CblasNoTrans,
            CblasNoTrans,
            batch_size,
            N,
            M,
            1,
            0,
            M,
            M * N,
            1.f,
            A.template data<T, Context>(),
            dY.template data<T, Context>(),
            0.f,
            dB->ReshapeLike(B)->template mutable_data<T, Context>(),
            ctx());
      }
    }
    return;
  }

  if (B_ndim == 1) {
    const auto N = B.count();
    const auto M = A.count() / N;
    // Matrix x Vector
    if (dA->has_name()) {
      math::Gemm(
          CblasNoTrans,
          CblasNoTrans,
          M,
          N,
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
          M,
          N,
          1.f,
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          0.f,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
    return;
  }

  // Check matrix A && B
  const auto M = A.dim(A_ndim - 2);
  const auto K = A.dim(A_ndim - 1);
  const auto N = B.dim(B_ndim - 1);

  // Check batching && broadcasting
  vec64_t A_dims(A.dims().begin(), A.dims().end() - 2);
  vec64_t B_dims(B.dims().begin(), B.dims().end() - 2);
  vec64_t A_batch_dims, B_batch_dims, Y_batch_dims;
  vec32_t A_batch_axes, B_batch_axes;
  if (math::utils::IsBinaryBroadcast(A_dims, B_dims, Y_batch_dims)) {
    math::utils::ComputeBinaryBroadcastDims(
        A_dims, B_dims, A_batch_dims, B_batch_dims);
    math::utils::ComputeBinaryBroadcastAxes(
        A_batch_dims, B_batch_dims, Y_batch_dims, A_batch_axes, B_batch_axes);
  } else {
    LOG(FATAL) << "Could not broadcast together with shapes " << A.DimString()
               << " " << B.DimString();
  }
  const int64_t batch_ndim = A_batch_dims.size();
  const bool broadcasting = A_batch_dims != B_batch_dims;

  const auto A_batch_size = std::accumulate(
      A_batch_dims.begin(),
      A_batch_dims.end(),
      1LL,
      std::multiplies<int64_t>());
  const auto B_batch_size = std::accumulate(
      B_batch_dims.begin(),
      B_batch_dims.end(),
      1LL,
      std::multiplies<int64_t>());
  const auto Y_batch_size = std::accumulate(
      Y_batch_dims.begin(),
      Y_batch_dims.end(),
      1LL,
      std::multiplies<int64_t>());

  if (B_batch_size == 1) {
    // Batched Matrix x Broadcasted Matrix
    if (dA->has_name()) {
      math::Gemm(
          CblasNoTrans,
          CblasTrans,
          A_batch_size * M,
          K,
          N,
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
          K,
          N,
          A_batch_size * M,
          1.f,
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          0.f,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  } else if (A_batch_size == 1) {
    // Broadcasted Matrix x Batched Matrix
    if (dA->has_name()) {
      auto* scratch = ctx()->workspace()->template data<T, Context>(
          {Y_batch_size * M * K})[0];
      math::GemmStridedBatched(
          CblasNoTrans,
          CblasTrans,
          Y_batch_size,
          M,
          K,
          N,
          M * N,
          K * N,
          M * K,
          1.f,
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          0.0f,
          scratch,
          ctx());
      math::ReduceSum(
          2,
          vec32_t{int(Y_batch_size), int(M * K)}.data(),
          1,
          vec32_t{0}.data(),
          1.f,
          scratch,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
    if (dB->has_name()) {
      math::GemmStridedBatched(
          CblasTrans,
          CblasNoTrans,
          Y_batch_size,
          K,
          N,
          M,
          0,
          M * N,
          K * N,
          1.f,
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          0.f,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  } else if (!broadcasting) {
    // Batched Matrix x Batched Matrix
    if (dA->has_name()) {
      math::GemmStridedBatched(
          CblasNoTrans,
          CblasTrans,
          Y_batch_size,
          M,
          K,
          N,
          M * N,
          K * N,
          M * K,
          1.f,
          dY.template data<T, Context>(),
          B.template data<T, Context>(),
          0.f,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
    if (dB->has_name()) {
      math::GemmStridedBatched(
          CblasTrans,
          CblasNoTrans,
          Y_batch_size,
          K,
          N,
          M,
          M * K,
          M * N,
          K * N,
          1.f,
          A.template data<T, Context>(),
          dY.template data<T, Context>(),
          0.f,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  } else {
    // Broadcasted Matrix x Broadcasted Matrix
    vector<const T*> A_arr(Y_batch_size);
    vector<const T*> B_arr(Y_batch_size);
    vector<const T*> dY_arr(Y_batch_size);
    vector<T*> dA_arr(Y_batch_size);
    vector<T*> dB_arr(Y_batch_size);
    if (dA->has_name()) {
      vec64_t index(batch_ndim, 0);
      vec32_t scratch_dims({Y_batch_dims.begin(), Y_batch_dims.end()});
      scratch_dims.push_back(int(M * K));
      auto* dY_data = dY.template data<T, Context>();
      auto* B_data = B.template data<T, Context>();
      auto* scratch = ctx()->workspace()->template data<T, Context>(
          {Y_batch_size * std::max(M * K, K * N)})[0];
      for (int64_t Y_i = 0; Y_i < Y_batch_size; ++Y_i) {
        const auto B_i = math::utils::GetIndexFromDims(
            batch_ndim, B_batch_dims.data(), index.data());
        dY_arr[Y_i] = dY_data + Y_i * M * N;
        B_arr[Y_i] = B_data + B_i * K * N;
        dA_arr[Y_i] = scratch + Y_i * M * K;
        math::utils::IncreaseIndexInDims(
            batch_ndim, Y_batch_dims.data(), index.data());
      }
      math::GemmBatched(
          CblasNoTrans,
          CblasTrans,
          Y_batch_size,
          M,
          K,
          N,
          1.f,
          dY_arr.data(),
          B_arr.data(),
          0.f,
          dA_arr.data(),
          ctx());
      math::ReduceSum(
          scratch_dims.size(),
          scratch_dims.data(),
          A_batch_axes.size(),
          A_batch_axes.data(),
          1.f,
          scratch,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
    if (dB->has_name()) {
      vec64_t index(batch_ndim, 0);
      vec32_t scratch_dims({Y_batch_dims.begin(), Y_batch_dims.end()});
      scratch_dims.push_back(int(K * N));
      auto* dY_data = dY.template data<T, Context>();
      auto* A_data = A.template data<T, Context>();
      auto* scratch = ctx()->workspace()->template data<T, Context>(
          {Y_batch_size * std::max(M * K, K * N)})[0];
      for (int64_t Y_i = 0; Y_i < Y_batch_size; ++Y_i) {
        const auto A_i = math::utils::GetIndexFromDims(
            batch_ndim, A_batch_dims.data(), index.data());
        dY_arr[Y_i] = dY_data + Y_i * M * N;
        A_arr[Y_i] = A_data + A_i * M * K;
        dB_arr[Y_i] = scratch + Y_i * K * N;
        math::utils::IncreaseIndexInDims(
            batch_ndim, Y_batch_dims.data(), index.data());
      }
      math::GemmBatched(
          CblasTrans,
          CblasNoTrans,
          Y_batch_size,
          K,
          N,
          M,
          1.f,
          A_arr.data(),
          dY_arr.data(),
          0.f,
          dB_arr.data(),
          ctx());
      math::ReduceSum(
          scratch_dims.size(),
          scratch_dims.data(),
          B_batch_axes.size(),
          B_batch_axes.data(),
          1.f,
          scratch,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
void MatMulGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(MatMul);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(MatMul);
#endif

DEPLOY_CPU_OPERATOR(MatMulGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(MatMulGradient);
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
