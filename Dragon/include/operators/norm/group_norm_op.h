/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_NORM_GROUP_NORM_OP_H_
#define DRAGON_OPERATORS_NORM_GROUP_NORM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class GroupNormOp final : public Operator<Context> {
 public:
    GroupNormOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", -1)),
          group_(OpArg<int64_t>("group", 0)),
          eps_(OpArg<float>("eps", 1e-5f)) {}
    USE_OPERATOR_FUNCTIONS;

    void Reshape();

    void RunOnDevice() override;
    template <typename Tx, typename Tp> void RunImpl();

 protected:
    float eps_;
    int64_t group_, axis_;
    int64_t N_, C_, G_, D_, S_;
    Tensor *mean_, *var_, scale_, bias_;
};

template <class Context>
class GroupNormGradientOp final : public Operator<Context> {
 public:
    GroupNormGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", -1)),
          group_(OpArg<int64_t>("group", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void Reshape();

    void RunOnDevice() override;
    template <typename Tx, typename Tp> void RunImpl();

 protected:
    int64_t group_, axis_;
    int64_t N_, C_, G_, D_, S_;
    Tensor *mean_, *var_, dscale_, dbias_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NORM_GROUP_NORM_OP_H_