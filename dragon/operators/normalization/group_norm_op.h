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

namespace dragon {

template <class Context>
class GroupNormOpBase : public Operator<Context> {
 public:
  GroupNormOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        group_(OP_SINGLE_ARG(int64_t, "group", 0)),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-5)) {}
  USE_OPERATOR_FUNCTIONS;

  void DetermineBaseArguments() {
    auto& X = Input(0);
    // Determine the data format
    this->data_format_ = "NCHW";
    auto axis = OP_SINGLE_ARG(int64_t, "axis", -1);
    if (axis == -1) axis += X.ndim();
    if (axis + 1 == X.ndim()) this->data_format_ = "NHWC";
    if (X.ndim() == 2) this->data_format_ = "NCHW";
    N_ = X.dim(0), C_ = X.dim(axis);
    S_ = X.count() / N_ / C_;
    // InstanceNorm, LayerNorm or GroupNorm ?
    G_ = group_ > 0 ? group_ : C_;
    D_ = C_ / G_;
    // Check the channels and groups
    CHECK_EQ(C_ % G_, 0) << "\nThe " << C_ << " channels "
                         << "can not be split into " << G_ << " groups.";
  }

 protected:
  double epsilon_;
  int64_t group_;
  int64_t N_, C_, G_, D_, S_;
};

#define USE_GROUPNORM_FUNCTIONS                           \
  using GroupNormOpBase<Context>::DetermineBaseArguments; \
  using GroupNormOpBase<Context>::group_;                 \
  using GroupNormOpBase<Context>::epsilon_;               \
  using GroupNormOpBase<Context>::N_;                     \
  using GroupNormOpBase<Context>::C_;                     \
  using GroupNormOpBase<Context>::G_;                     \
  using GroupNormOpBase<Context>::D_;                     \
  using GroupNormOpBase<Context>::S_

template <class Context>
class GroupNormOp final : public GroupNormOpBase<Context> {
 public:
  GroupNormOp(const OperatorDef& def, Workspace* ws)
      : GroupNormOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_GROUPNORM_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class GroupNormGradientOp final : public GroupNormOpBase<Context> {
 public:
  GroupNormGradientOp(const OperatorDef& def, Workspace* ws)
      : GroupNormOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_GROUPNORM_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_GROUP_NORM_OP_H_
