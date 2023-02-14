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

#ifndef DRAGON_OPERATORS_NORMALIZATION_GROUP_NORM_OP_H_
#define DRAGON_OPERATORS_NORMALIZATION_GROUP_NORM_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/math/reduce_op_impl_cnnl.h"

namespace dragon {

template <class Context>
class GroupNormOpBase : public Operator<Context> {
 public:
  GroupNormOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-5)) {}
  USE_OPERATOR_FUNCTIONS;

  virtual void GetBaseArguments() {
    auto& X = Input(0);
    GET_OP_AXIS_ARG(axis, X.ndim(), -1);
    N_ = X.dim(0);
    C_ = X.dim(axis);
    S_ = X.count() / N_ / C_;
    G_ = OP_SINGLE_ARG(int64_t, "group", 0);
    G_ = G_ > 0 ? G_ : C_;
    D_ = C_ / G_;
    CHECK_EQ(C_ % G_, 0) << "\nThe " << C_ << " channels "
                         << "can not be split into " << G_ << " groups.";
    this->data_format_ = "NCHW";
    if (axis + 1 == X.ndim()) this->data_format_ = "NHWC";
  }

 protected:
  double epsilon_;
  int64_t N_, C_, G_, D_, S_;
};

#define USE_GROUPNORM_FUNCTIONS                     \
  using GroupNormOpBase<Context>::GetBaseArguments; \
  using GroupNormOpBase<Context>::epsilon_;         \
  using GroupNormOpBase<Context>::N_;               \
  using GroupNormOpBase<Context>::C_;               \
  using GroupNormOpBase<Context>::G_;               \
  using GroupNormOpBase<Context>::D_;               \
  using GroupNormOpBase<Context>::S_

template <class Context>
class GroupNormOp : public GroupNormOpBase<Context> {
 public:
  GroupNormOp(const OperatorDef& def, Workspace* ws)
      : GroupNormOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_GROUPNORM_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class GroupNormGradientOp : public GroupNormOpBase<Context> {
 public:
  GroupNormGradientOp(const OperatorDef& def, Workspace* ws)
      : GroupNormOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_GROUPNORM_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MLU
template <class Context>
class CNNLGroupNormOp : public GroupNormOpBase<Context> {
 public:
  CNNLGroupNormOp(const OperatorDef& def, Workspace* ws)
      : GroupNormOpBase<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&scale_desc_);
    CNNLCreateTensorDesc(&stats_desc_);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_GROUPNORM_FUNCTIONS;

  ~CNNLGroupNormOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(scale_desc_);
    CNNLDestroyTensorDesc(stats_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlTensorDescriptor_t input_desc_, scale_desc_, stats_desc_;
};

template <class Context>
class CNNLGroupNormGradientOp final : public CNNLGroupNormOp<Context> {
 public:
  CNNLGroupNormGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLGroupNormOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_GROUPNORM_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  CNNLReduceOpImpl mean_impl_, rsig_impl_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_GROUP_NORM_OP_H_
