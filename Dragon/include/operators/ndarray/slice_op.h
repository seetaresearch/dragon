// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_NDARRAY_SLICE_OP_H_
#define DRAGON_OPERATORS_NDARRAY_SLICE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SliceOp : public Operator<Context> {
 public:
    SliceOp(const OperatorDef& op_def, Workspace* ws):
        Operator<Context>(op_def, ws),
        axis(OperatorBase::GetSingleArg<int>("axis", 1)),
        nout(OperatorBase::GetSingleArg<int>("num_output", 1)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, nout, steps;
    TIndex outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    TIndex slice_offset;
    vector<TIndex> slice_dims;
};

template <class Context>
class SliceGradientOp final : public Operator<Context> {
 public:
    SliceGradientOp(const OperatorDef& op_def, Workspace* ws):
        Operator<Context>(op_def, ws),
        axis(OperatorBase::GetSingleArg<int>("axis", 1)),
        nout(OperatorBase::GetSingleArg<int>("num_output", 1)) {
        DISABLE_SHARE_GRADIENT;
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, nout;
    TIndex outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    TIndex x_offset, y_offset, slice_offset;
    vector<TIndex> slice_dims;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_SLICE_OP_H_