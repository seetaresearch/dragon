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

#ifndef DRAGON_OPERATORS_ARRAY_SLICE_OP_H_
#define DRAGON_OPERATORS_ARRAY_SLICE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SliceOp final : public Operator<Context> {
 public:
     SliceOp(const OperatorDef& def, Workspace* ws)
         : Operator<Context>(def, ws),
           axis(OperatorBase::Arg<int64_t>("axis", 0)),
           slice_points(OperatorBase::Args<int64_t>("slice_points")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int64_t axis, N, steps, slice_offset;
    int64_t outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    vector<int64_t> slice_dims, slice_points;
};

template <class Context>
class SliceGradientOp final : public Operator<Context> {
 public:
    SliceGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 0)),
          slice_points(OperatorBase::Args<int64_t>("slice_points")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int64_t axis, N, x_offset, y_offset, slice_offset;
    int64_t outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    vector<int64_t> slice_dims, slice_points;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_SLICE_OP_H_