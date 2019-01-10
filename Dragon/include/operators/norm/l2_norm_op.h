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

#ifndef DRAGON_OPERATORS_NORM_L2_NORM_H_
#define DRAGON_OPERATORS_NORM_L2_NORM_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class L2NormOp final : public Operator<Context> {
 public:
    L2NormOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 0)),
          num_axes(OperatorBase::Arg<int64_t>("num_axes", -1)),
          eps(OperatorBase::Arg<float>("eps", 1e-5f)),
          mode(OperatorBase::Arg<string>("mode", "SUM")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float eps; string mode;
    int64_t axis, num_axes, end_axis;
    int64_t outer_dim, dim, inner_dim, spatial_dim;
    Tensor* norm, buffer;
};

template <class Context>
class L2NormGradientOp final : public Operator<Context> {
 public:
    L2NormGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 0)),
          num_axes(OperatorBase::Arg<int64_t>("num_axes", -1)),
          mode(OperatorBase::Arg<string>("mode", "SUM")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    string mode;
    int64_t axis, num_axes, end_axis;
    int64_t outer_dim, dim, inner_dim;
    Tensor* norm, buffer, buffer_inner;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NORM_L2_NORM_H_