// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

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
        axis(OperatorBase::GetSingleArg<int>("axis", 0)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

protected:
    TIndex axis;
    TIndex outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    Tensor* pick_indices;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_RANDOM_PICK_OP_H_