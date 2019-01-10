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
          axis(OperatorBase::Arg<int64_t>("axis", -1)),
          group(OperatorBase::Arg<int64_t>("group", 0)),
          eps(OperatorBase::Arg<float>("eps", 1e-5f)) {}
    USE_OPERATOR_FUNCTIONS;

    void Reshape();

    void RunOnDevice() override;
    template <typename Tx, typename Tp> void RunWithType();

 protected:
    float eps;
    string data_format;
    int64_t group, axis, N, C, G, D, S;
    Tensor *mean, *var, scale, bias;
};

template <class Context>
class GroupNormGradientOp final : public Operator<Context> {
 public:
    GroupNormGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", -1)),
          group(OperatorBase::Arg<int64_t>("group", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void Reshape();

    void RunOnDevice() override;
    template <typename Tx, typename Tp> void RunWithType();

 protected:
    int64_t group, axis, N, C, G, D, S;
    Tensor *mean, *var, dscale, dbias;
    string data_format;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NORM_GROUP_NORM_OP_H_