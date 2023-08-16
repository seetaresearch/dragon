/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_MATH_GEMM_OP_IMPL_CUDA_H_
#define DRAGON_OPERATORS_MATH_GEMM_OP_IMPL_CUDA_H_

#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

constexpr int CUBLASLT_MATMUL_ALGO_COUNT = 5;

template <typename Algo>
class CUDAGemmOpImpl {};

template <typename T>
cudaDataType_t CUDAGetGemmDataType() {
  static std::unordered_map<TypeId, cudaDataType_t> m{
      {TypeMeta::Id<float16>(), CUDA_R_16F},
      {TypeMeta::Id<bfloat16>(), CUDA_R_16BF},
      {TypeMeta::Id<float>(), CUDA_R_32F},
      {TypeMeta::Id<double>(), CUDA_R_64F},
  };
  auto it = m.find(TypeMeta::Id<T>());
  CHECK(it != m.end()) << "\nUnsupported " << dtypes::to_string<T>()
                       << " GEMM.";
  return it->second;
}

template <>
class CUDAGemmOpImpl<cublasLtMatmulAlgo_t> {
 public:
  CUDAGemmOpImpl() : scratch_size_(0) {
    CUBLAS_CHECK(
        cublasLtMatmulDescCreate(&mm_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_32F, 1, 1, 1));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_32F, 1, 1, 1));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Y_desc_, CUDA_R_32F, 1, 1, 1));
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&perf_));
  }

  ~CUDAGemmOpImpl() {
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(A_desc_));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(B_desc_));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Y_desc_));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(mm_desc_));
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(perf_));
  }

  template <typename T>
  void SetEpilogueAndBias(
      const cublasLtEpilogue_t epilogue,
      T* bias,
      void* aux = nullptr,
      const int64_t aux_ld = 1) {
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        mm_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        mm_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        mm_desc_,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
        &aux,
        sizeof(aux)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        mm_desc_,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
        &aux_ld,
        sizeof(aux_ld)));
  }

  template <typename T>
  void Setup(
      const int64_t TransA,
      const int64_t TransB,
      const float alpha,
      const float beta,
      const vec64_t& A_dims,
      const vec64_t& B_dims,
      CUDAContext* ctx) {
    const auto cuTransA = TransA == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto cuTransB = TransB == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto M = math::utils::Prod(A_dims.size() - 1, A_dims.data());
    auto K2 = math::utils::Prod(B_dims.size() - 1, B_dims.data());
    auto K1 = A_dims.back(), N = B_dims.back();
    if (TransA > 0) std::swap(M, K1);
    if (TransB > 0) std::swap(K2, N);
    const auto& ABY_dtype = CUDAGetGemmDataType<T>();
    const auto& ABY_dprop = CUDAGetDeviceProp(ctx->device());
    const auto is_fp64 = ABY_dtype == CUDA_R_64F;
    const auto compute_type = is_fp64 ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
    const auto scale_type = ABY_dtype == CUDA_R_64F ? CUDA_R_64F : CUDA_R_32F;
    beta32f_ = beta, beta64f_ = double(beta);
    alpha32f_ = alpha, alpha64f_ = double(alpha);
    beta_ptr_ = is_fp64 ? (float*)&beta64f_ : &beta32f_;
    alpha_ptr_ = is_fp64 ? (float*)&alpha64f_ : &alpha32f_;
    scratch_size_ = 1024 * 1024 * (ABY_dprop.major >= 9 ? 32 : 4);
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        mm_desc_,
        CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
        &compute_type,
        sizeof(compute_type)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        mm_desc_,
        CUBLASLT_MATMUL_DESC_SCALE_TYPE,
        &scale_type,
        sizeof(scale_type)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        mm_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &cuTransA, sizeof(cuTransA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        mm_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &cuTransB, sizeof(cuTransB)));
    CUBLAS_CHECK(cublasLtMatrixLayoutInit(
        A_desc_,
        ABY_dtype,
        TransA == 0 ? K1 : M,
        TransA == 0 ? M : K1,
        TransA == 0 ? K1 : M));
    CUBLAS_CHECK(cublasLtMatrixLayoutInit(
        B_desc_,
        ABY_dtype,
        TransB == 0 ? N : K2,
        TransB == 0 ? K2 : N,
        TransB == 0 ? N : K2));
    CUBLAS_CHECK(cublasLtMatrixLayoutInit(Y_desc_, ABY_dtype, N, M, N));
    CUBLAS_CHECK(cublasLtMatmulPreferenceInit(perf_));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        perf_,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &scratch_size_,
        sizeof(scratch_size_)));
    int num_valid;
    cublasLtMatmulHeuristicResult_t algo_res[CUBLASLT_MATMUL_ALGO_COUNT];
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        reinterpret_cast<cublasLtHandle_t>(ctx->cublas_handle()),
        mm_desc_,
        B_desc_,
        A_desc_,
        Y_desc_,
        Y_desc_,
        perf_,
        CUBLASLT_MATMUL_ALGO_COUNT,
        algo_res,
        &num_valid));
    CHECK_GT(num_valid, 0) << "\nFailed to get <cublasLtMatmul> algos.";
    algo_ = algo_res[0].algo;
  }

  template <typename T>
  void Compute(const T* A, const T* B, T* Y, void* scratch, CUDAContext* ctx) {
    CUBLAS_CHECK(cublasLtMatmul(
        reinterpret_cast<cublasLtHandle_t>(ctx->cublas_handle()),
        mm_desc_,
        alpha_ptr_,
        B,
        B_desc_,
        A,
        A_desc_,
        beta_ptr_,
        Y,
        Y_desc_,
        Y,
        Y_desc_,
        &algo_,
        scratch,
        scratch_size_,
        ctx->cuda_stream()));
  }

  size_t scratch_size() {
    return scratch_size_;
  }

 private:
  float alpha32f_, beta32f_;
  double alpha64f_, beta64f_;
  void *alpha_ptr_, *beta_ptr_;
  size_t scratch_size_;
  cublasLtMatmulAlgo_t algo_;
  cublasLtMatmulPreference_t perf_;
  cublasLtMatmulDesc_t mm_desc_;
  cublasLtMatrixLayout_t A_desc_, B_desc_, Y_desc_;
};

} // namespace dragon

#endif // USE_CUDA

#endif // DRAGON_OPERATORS_MATH_GEMM_OP_IMPL_CUDA_H_
