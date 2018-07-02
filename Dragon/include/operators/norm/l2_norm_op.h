// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_NORM_L2_NORM_H_
#define DRAGON_OPERATORS_NORM_L2_NORM_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class L2NormOp final : public Operator<Context> {
 public:
    L2NormOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 0)),
          num_axes(OperatorBase::Arg<int>("num_axes", -1)),
          eps(OperatorBase::Arg<float>("eps", 1e-5f)),
          mode(OperatorBase::Arg<string>("mode", "SUM")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, num_axes, end_axis;
    float eps;
    string mode;
    bool across_inner;
    Tensor* norm, buffer;
    TIndex outer_dim, dim, inner_dim, spatial_dim;
};

template <class Context>
class L2NormGradientOp final : public Operator<Context> {
 public:
    L2NormGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 0)),
          num_axes(OperatorBase::Arg<int>("num_axes", -1)),
          mode(OperatorBase::Arg<string>("mode", "SUM")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, num_axes, end_axis;
    string mode;
    bool across_inner;
    Tensor* norm, buffer, buffer_inner;
    TIndex outer_dim, dim, inner_dim;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NORM_L2_NORM_H_