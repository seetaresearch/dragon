// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_COMMON_CONCAT_OP_H_
#define DRAGON_OPERATORS_COMMON_CONCAT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ConcatOp : public Operator<Context> {
 public:
    ConcatOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)),
          nin(OperatorBase::GetSingleArg<int>("num_input", 1)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, nin, outer_dim, inner_dim, x_concat_dim, y_concat_dim;
    TIndex x_offset, y_offset, concat_offset;
    vector<TIndex> concat_dims;
};

template <class Context>
class ConcatGradientOp : public Operator<Context> {
 public:
    ConcatGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)),
          nin(OperatorBase::GetSingleArg<int>("num_input", 1)) {}

    void ShareBeforeRun() override;
    void RunOnDevice() override;
    void ClearAfterRun() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, nin, outer_dim, inner_dim, x_concat_dim, y_concat_dim;
    TIndex x_offset, y_offset, concat_offset;
    vector<TIndex> concat_dims;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_COMMON_CONCAT_OP_H_