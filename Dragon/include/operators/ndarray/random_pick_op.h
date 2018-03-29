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

#ifndef DRAGON_OPERATORS_NDARRAY_RANDOM_PICK_OP_H_
#define DRAGON_OPERATORS_NDARRAY_RANDOM_PICK_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class RandomPickOp : public Operator<Context> {
 public:
    RandomPickOp(const OperatorDef& op_def, Workspace* ws) :
        Operator<Context>(op_def, ws),
        axis(OperatorBase::GetSingleArg<int>("axis", 0)),
        max_samples(OperatorBase::GetSingleArg<int>("max_samples", 1)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, max_samples;
    TIndex outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    vector<TIndex> output_dims;
    Tensor* pick_indices;
};

template <class Context>
class RandomPickGradientOp final : public Operator<Context> {
public:
    RandomPickGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
        axis(OperatorBase::GetSingleArg<int>("axis", 0)) {
        DISABLE_SHARE_GRADIENT;
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

protected:
    TIndex axis;
    TIndex outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    Tensor* pick_indices;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_RANDOM_PICK_OP_H_