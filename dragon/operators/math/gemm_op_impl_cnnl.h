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

#ifndef DRAGON_OPERATORS_MATH_GEMM_OP_IMPL_CNNL_H_
#define DRAGON_OPERATORS_MATH_GEMM_OP_IMPL_CNNL_H_

#ifdef USE_MLU

#include "dragon/core/context_mlu.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <typename Algo>
class CNNLGemmOpImpl {};

template <typename Algo>
class CNNLBatchGemmOpImpl {};

template <>
class CNNLGemmOpImpl<cnnlMatMulAlgo_t> {
 public:
  CNNLGemmOpImpl() : scratch_size_(0) {
    CNNLCreateTensorDesc(&A_desc_);
    CNNLCreateTensorDesc(&B_desc_);
    CNNLCreateTensorDesc(&Y_desc_);
    CNNL_CHECK(cnnlMatMulDescCreate(&mm_desc_));
    CNNL_CHECK(cnnlMatMulAlgoCreate(&algo_));
    CNNL_CHECK(cnnlCreateMatMulHeuristicResult(&perf_));
    const auto& compute_type = CNNLGetDataType<float>();
    CNNL_CHECK(cnnlSetMatMulDescAttr(
        mm_desc_,
        CNNL_MATMUL_DESC_COMPUTE_TYPE,
        &compute_type,
        sizeof(compute_type)));
  }

  ~CNNLGemmOpImpl() {
    CNNLDestroyTensorDesc(A_desc_);
    CNNLDestroyTensorDesc(B_desc_);
    CNNLDestroyTensorDesc(Y_desc_);
    CNNL_CHECK(cnnlMatMulDescDestroy(mm_desc_));
    CNNL_CHECK(cnnlMatMulAlgoDestroy(algo_));
    CNNL_CHECK(cnnlDestroyMatMulHeuristicResult(perf_));
  }

  template <typename T>
  void Setup(
      const int64_t TransA,
      const int64_t TransB,
      const float alpha,
      const float beta,
      const vec64_t& A_dims,
      const vec64_t& B_dims,
      MLUContext* ctx) {
    const int transA = TransA, transB = TransB;
    auto M = math::utils::Prod(A_dims.size() - 1, A_dims.data());
    auto K2 = math::utils::Prod(B_dims.size() - 1, B_dims.data());
    auto K1 = A_dims.back(), N = B_dims.back();
    const vec64_t A_squeeze_dims({M, K1});
    const vec64_t B_squeeze_dims({K2, N});
    if (TransA > 0) std::swap(M, K1);
    if (TransB > 0) std::swap(K2, N);
    alpha_ = alpha, beta_ = beta;
    const auto use_beta = beta != 0.f;
    CNNLSetTensorDesc<T>(A_desc_, A_squeeze_dims);
    CNNLSetTensorDesc<T>(B_desc_, B_squeeze_dims);
    CNNLSetTensorDesc<T>(Y_desc_, vec64_t({M, N}));
    CNNL_CHECK(cnnlSetMatMulDescAttr(
        mm_desc_, CNNL_MATMUL_DESC_TRANSA, &transA, sizeof(int)));
    CNNL_CHECK(cnnlSetMatMulDescAttr(
        mm_desc_, CNNL_MATMUL_DESC_TRANSB, &transB, sizeof(int)));
    CNNL_CHECK(cnnlSetMatMulDescAttr(
        mm_desc_, CNNL_MATMUL_USE_BETA, &use_beta, sizeof(use_beta)));
    int num_valid;
    CNNL_CHECK(cnnlGetMatMulAlgoHeuristic(
        ctx->cnnl_handle(),
        mm_desc_,
        A_desc_,
        B_desc_,
        Y_desc_,
        Y_desc_,
        nullptr,
        1,
        &perf_,
        &num_valid));
    CNNL_CHECK(cnnlGetMatMulHeuristicResult(perf_, algo_, &scratch_size_));
  }

  template <typename T>
  void Compute(const T* A, const T* B, T* Y, void* scratch, MLUContext* ctx) {
    CNNL_CHECK(cnnlMatMul_v2(
        ctx->cnnl_handle(),
        mm_desc_,
        algo_,
        &alpha_,
        A_desc_,
        A,
        B_desc_,
        B,
        &beta_,
        Y_desc_,
        Y,
        scratch,
        scratch_size_,
        Y_desc_,
        Y));
  }

  size_t scratch_size() {
    return scratch_size_;
  }

 private:
  float alpha_, beta_;
  size_t scratch_size_;
  cnnlMatMulAlgo_t algo_;
  cnnlMatMulHeuristicResult_t perf_;
  cnnlMatMulDescriptor_t mm_desc_;
  cnnlTensorDescriptor_t A_desc_, B_desc_, Y_desc_;
};

template <>
class CNNLBatchGemmOpImpl<cnnlMatMulAlgo_t> {
 public:
  CNNLBatchGemmOpImpl() {
    CNNLCreateTensorDesc(&A_desc_);
    CNNLCreateTensorDesc(&B_desc_);
    CNNLCreateTensorDesc(&Y_desc_);
    CNNL_CHECK(cnnlMatMulDescCreate(&mm_desc_));
    CNNL_CHECK(cnnlMatMulAlgoCreate(&algo_));
    CNNL_CHECK(cnnlCreateMatMulHeuristicResult(&perf_));
    const auto& compute_type = CNNLGetDataType<float>();
    CNNL_CHECK(cnnlSetMatMulDescAttr(
        mm_desc_,
        CNNL_MATMUL_DESC_COMPUTE_TYPE,
        &compute_type,
        sizeof(compute_type)));
  }

  ~CNNLBatchGemmOpImpl() {
    CNNLDestroyTensorDesc(A_desc_);
    CNNLDestroyTensorDesc(B_desc_);
    CNNLDestroyTensorDesc(Y_desc_);
    CNNL_CHECK(cnnlMatMulDescDestroy(mm_desc_));
    CNNL_CHECK(cnnlMatMulAlgoDestroy(algo_));
    CNNL_CHECK(cnnlDestroyMatMulHeuristicResult(perf_));
  }

  template <typename T>
  void Setup(
      const int64_t TransA,
      const int64_t TransB,
      const float alpha,
      const float beta,
      const vec64_t& A_dims,
      const vec64_t& B_dims,
      MLUContext* ctx) {
    const int transA = TransA, transB = TransB;
    auto M = *(A_dims.end() - 2), K1 = A_dims.back();
    auto K2 = *(B_dims.end() - 2), N = B_dims.back();
    vec64_t A_bcast_dims, B_bcast_dims, Y_bcast_dims;
    vec64_t A_batch_dims(A_dims.begin(), A_dims.end() - 2);
    vec64_t B_batch_dims(B_dims.begin(), B_dims.end() - 2);
    math::utils::ComputeBroadcastDims(
        A_batch_dims, B_batch_dims, A_bcast_dims, B_bcast_dims);
    for (int i = 0; i < A_bcast_dims.size(); ++i) {
      Y_bcast_dims.push_back(std::max(A_bcast_dims[i], B_bcast_dims[i]));
    }
    A_bcast_dims.insert(A_bcast_dims.end(), A_dims.end() - 2, A_dims.end());
    B_bcast_dims.insert(B_bcast_dims.end(), B_dims.end() - 2, B_dims.end());
    if (TransA > 0) std::swap(M, K1);
    if (TransB > 0) std::swap(K2, N);
    Y_bcast_dims.push_back(M);
    Y_bcast_dims.push_back(N);
    alpha_ = alpha, beta_ = beta;
    const bool use_beta = beta != 0.f;
    CNNLSetTensorDesc<T>(A_desc_, A_bcast_dims);
    CNNLSetTensorDesc<T>(B_desc_, B_bcast_dims);
    CNNLSetTensorDesc<T>(Y_desc_, Y_bcast_dims);
    CNNL_CHECK(cnnlSetMatMulDescAttr(
        mm_desc_, CNNL_MATMUL_DESC_TRANSA, &transA, sizeof(int)));
    CNNL_CHECK(cnnlSetMatMulDescAttr(
        mm_desc_, CNNL_MATMUL_DESC_TRANSB, &transB, sizeof(int)));
    CNNL_CHECK(cnnlSetMatMulDescAttr(
        mm_desc_, CNNL_MATMUL_USE_BETA, &use_beta, sizeof(use_beta)));
    int num_valid;
    CNNL_CHECK(cnnlGetBatchMatMulAlgoHeuristic(
        ctx->cnnl_handle(),
        mm_desc_,
        A_desc_,
        B_desc_,
        Y_desc_,
        nullptr,
        1,
        &perf_,
        &num_valid));
    CNNL_CHECK(cnnlGetBatchMatMulHeuristicResult(perf_, algo_, &scratch_size_));
  }

  template <typename T>
  void Compute(const T* A, const T* B, T* Y, void* scratch, MLUContext* ctx) {
    CNNL_CHECK(cnnlBatchMatMulBCast_v2(
        ctx->cnnl_handle(),
        mm_desc_,
        algo_,
        &alpha_,
        A_desc_,
        A,
        B_desc_,
        B,
        &beta_,
        Y_desc_,
        Y,
        scratch,
        scratch_size_));
  }

  size_t scratch_size() {
    return scratch_size_;
  }

 private:
  float alpha_, beta_;
  size_t scratch_size_;
  cnnlMatMulAlgo_t algo_;
  cnnlMatMulHeuristicResult_t perf_;
  cnnlMatMulDescriptor_t mm_desc_;
  cnnlTensorDescriptor_t A_desc_, B_desc_, Y_desc_;
};

} // namespace dragon

#endif // USE_MLU

#endif // DRAGON_OPERATORS_MATH_GEMM_OP_IMPL_CNNL_H_
