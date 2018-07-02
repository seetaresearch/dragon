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

#ifndef DRAGON_OPERATORS_NDARRAY_GATHER_OP_H_
#define DRAGON_OPERATORS_NDARRAY_GATHER_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class GatherOp final : public Operator<Context> {
 public:
    GatherOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    vector<TIndex> output_dims;
};

template <class Context>
class GatherGradientOp final : public Operator<Context> {
 public:
    GatherGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 0)),
          acc_grad(OperatorBase::Arg<bool>("acc_gradient", false)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    bool acc_grad;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_GATHER_OP_H_