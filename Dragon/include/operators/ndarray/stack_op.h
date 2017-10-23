// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_NDARRAY_STACK_OP_H_
#define DRAGON_OPERATORS_NDARRAY_STACK_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class StackOp : public Operator<Context> {
 public:
    StackOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 0)),
          nin(OperatorBase::GetSingleArg<int>("num_input", 1)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, nin, outer_dim, inner_dim, x_concat_dim, y_concat_dim;
    TIndex x_offset, y_offset, concat_offset;
    vector<TIndex> stack_dims, concat_dims;
};

template <class Context>
class StackGradientOp : public Operator<Context> {
 public:
    StackGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 0)),
          nin(OperatorBase::GetSingleArg<int>("num_input", 1)) {}

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, nin, outer_dim, inner_dim, x_concat_dim, y_concat_dim;
    TIndex x_offset, y_offset, concat_offset;
    vector<TIndex> concat_dims;
}; 

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_STACK_OP_H_