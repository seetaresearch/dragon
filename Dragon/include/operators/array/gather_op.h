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

#ifndef DRAGON_OPERATORS_ARRAY_GATHER_OP_H_
#define DRAGON_OPERATORS_ARRAY_GATHER_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class GatherOp final : public Operator<Context> {
 public:
    GatherOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int64_t axis, outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    vector<int64_t> output_dims;
};

template <class Context>
class GatherGradientOp final : public Operator<Context> {
 public:
    GatherGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 0)),
          zero_grad(OperatorBase::Arg<bool>("zero_grad", true)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    bool zero_grad;
    int64_t axis, outer_dim, inner_dim, x_slice_dim, y_slice_dim;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_GATHER_OP_H_