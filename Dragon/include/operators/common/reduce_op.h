// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_COMMON_REDUCE_OP_H_
#define DRAGON_OPERATORS_COMMON_REDUCE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ReduceOp final : public Operator<Context> {
 public:
    ReduceOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          operation(OperatorBase::GetSingleArg<string>("operation", "NONE")),
          keep_dims(OperatorBase::GetSingleArg<bool>("keep_dims", false)) {}

    void RunOnDevice() override;
    template <typename T> void SumRunWithType();
    template <typename T> void MeanRunWithType();

 protected:
    bool keep_dims;
    string operation;
    TIndex axis, axis_dim, count, inner_dim;
    Tensor* multiplier;
};

template <class Context>
class ReduceGradientOp final : public Operator<Context> {
 public:
    ReduceGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
        axis(OperatorBase::GetSingleArg<int>("axis", -1)),
        operation(OperatorBase::GetSingleArg<string>("operation", "NONE")) {}

    void ShareBeforeRun() override;
    void RunOnDevice() override;
    void ClearAfterRun() override;
    template <typename T> void SumRunWithType();
    template <typename T> void MeanRunWithType();

 protected:
    string operation;
    TIndex axis, axis_dim, count, inner_dim;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_COMMON_REDUCE_OP_H_