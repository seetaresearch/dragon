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

#ifndef DRAGON_OPERATORS_MATH_MATMUL_OP_H_
#define DRAGON_OPERATORS_MATH_MATMUL_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/math/gemm_op_impl_cnnl.h"
#include "dragon/operators/math/reduce_op_impl_cnnl.h"

namespace dragon {

template <class Context>
class MatMulOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(MatMulOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class MatMulGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(MatMulGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MPS
template <class Context>
class MPSMatMulOp final : public Operator<Context> {
 public:
  MPSMatMulOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), graph_(MPSCreateGraph()) {}
  USE_OPERATOR_FUNCTIONS;

  ~MPSMatMulOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

  template <typename T>
  void DoRunGraphWithType(
      const vector<MPSGraphTensor_t>& placeholders,
      const vec64_t& Y_dims);

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSMatMulGradientOp final : public Operator<Context> {
 public:
  MPSMatMulGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), graph_(MPSCreateGraph()) {}
  USE_OPERATOR_FUNCTIONS;

  ~MPSMatMulGradientOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};
#endif // USE_MPS

#ifdef USE_MLU
template <class Context>
class CNNLMatMulOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(CNNLMatMulOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  CNNLGemmOpImpl<cnnlMatMulAlgo_t> mm_impl_;
  CNNLBatchGemmOpImpl<cnnlMatMulAlgo_t> bmm_impl_;
};

template <class Context>
class CNNLMatMulGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(CNNLMatMulGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  CNNLGemmOpImpl<cnnlMatMulAlgo_t> mm_impl_;
  CNNLBatchGemmOpImpl<cnnlMatMulAlgo_t> bmm_impl_;
  CNNLReduceOpImpl reduce_impl_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_MATMUL_OP_H_
