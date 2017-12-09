// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_LOSS_SOFTMAX_CROSS_ENTROPY_OP_H_
#define DRAGON_OPERATORS_LOSS_SOFTMAX_CROSS_ENTROPY_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SoftmaxCrossEntropyOp final : public Operator<Context> {
 public:
    SoftmaxCrossEntropyOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)),
          normalization(OperatorBase::GetSingleArg<string>("normalization", "FULL")) {
        OperatorDef softmax_def = MakeOperatorDef("Softmax", "",
            vector<string>({ input(0).name() }),
            vector<string>({ "/mnt/" + anchor() + "/softmax_prob" }));
        softmax_def.add_arg()->CopyFrom(this->arg("axis"));
        if (op_def.has_device_option())
            softmax_def.mutable_device_option()->CopyFrom(op_def.device_option());
        softmax_op.reset(CreateOperator(softmax_def, ws));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim;
    Tensor losses;
    Tensor* prob;
    unique_ptr<OperatorBase> softmax_op;
    string normalization;
};

template <class Context>
class SoftmaxCrossEntropyGradientOp final : public Operator<Context> {
 public:
    SoftmaxCrossEntropyGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)),
          normalization(OperatorBase::GetSingleArg<string>("normalization", "FULL")) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim;
    Tensor* prob;
    string normalization;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_LOSS_SOFTMAX_CROSS_ENTROPY_OP_H_