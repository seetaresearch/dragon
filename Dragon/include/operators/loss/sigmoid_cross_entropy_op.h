// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_LOSS_SIGMOID_CROSS_ENTROPY_OP_H_
#define DRAGON_OPERATORS_LOSS_SIGMOID_CROSS_ENTROPY_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SigmoidCrossEntropyOp final : public Operator<Context> {
 public:
    SigmoidCrossEntropyOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          normalization(OperatorBase::GetSingleArg<string>("normalization", "FULL")) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor losses;
    Tensor* prob;
    string normalization;
};

template <class Context>
class SigmoidCrossEntropyGradientOp final : public Operator<Context> {
 public:
    SigmoidCrossEntropyGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          normalization(OperatorBase::GetSingleArg<string>("normalization", "FULL")) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor* prob;
    string normalization;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_LOSS_SIGMOID_CROSS_ENTROPY_OP_H_