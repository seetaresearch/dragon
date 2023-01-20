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

#ifndef DRAGON_OPERATORS_NORMALIZATION_LP_NORM_OP_H_
#define DRAGON_OPERATORS_NORMALIZATION_LP_NORM_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/math/reduce_op_impl_cnnl.h"

namespace dragon {

template <class Context>
class LpNormOp final : public Operator<Context> {
 public:
  LpNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OP_SINGLE_ARG(int64_t, "p", 2)),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-12)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "SUM")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t p_;
  double epsilon_;
  string reduction_;
};

template <class Context>
class LpNormGradientOp final : public Operator<Context> {
 public:
  LpNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OP_SINGLE_ARG(int64_t, "p", 2)),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-12)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "SUM")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t p_;
  float epsilon_;
  string reduction_;
};

#ifdef USE_MLU
template <class Context>
class CNNLLpNormOp final : public Operator<Context> {
 public:
  CNNLLpNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OP_SINGLE_ARG(int64_t, "p", 2)),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-12)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "SUM")) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&stats_desc_);
    CNNL_CHECK(cnnlCreateNormalizeDescriptor(&norm_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLLpNormOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(stats_desc_);
    CNNL_CHECK(cnnlDestroyNormalizeDescriptor(norm_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t p_;
  double epsilon_;
  string reduction_;
  cnnlTensorDescriptor_t input_desc_, stats_desc_;
  cnnlNormalizeDescriptor_t norm_desc_;
};

template <class Context>
class CNNLLpNormGradientOp final : public Operator<Context> {
 public:
  CNNLLpNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OP_SINGLE_ARG(int64_t, "p", 2)),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-12)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "SUM")) {
    reduce_impl_.SetReducer(CNNL_REDUCE_ADD);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t p_;
  float epsilon_;
  string reduction_;
  CNNLReduceOpImpl reduce_impl_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_LP_NORM_OP_H_
