// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_LOSS_SOFTMAX_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_SOFTMAX_LOSS_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SoftmaxLossOp final : public Operator<Context> {
 public:
    SoftmaxLossOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)),
          normalization(OperatorBase::GetSingleArg<string>("normalization", "VALID")) {
        vector<int> args = OperatorBase::GetRepeatedArg<int>("ignore_labels");
        if (args.size()) {
            ignore.Reshape(vector<TIndex>(1, args.size()));
            int* ignore_data = ignore.mutable_data<int, CPUContext>();
            for (int i = 0; i < args.size(); i++) ignore_data[i] = args[i];
        }
        OperatorDef softmax_def = MakeOperatorDef("Softmax", "",
            vector<string>({ input(0).name() }),
            vector<string>({ "_t_" + anchor() + "_softmax_prob" }));
        softmax_def.add_arg()->CopyFrom(this->arg("axis"));
        if (op_def.has_device_option())
            softmax_def.mutable_device_option()->CopyFrom(op_def.device_option());
        softmax_op.reset(CreateOperator(softmax_def, ws));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim;
    Tensor ignore, valid, losses;
    Tensor* prob;
    unique_ptr<OperatorBase> softmax_op;
    string normalization;
};

template <class Context>
class SoftmaxLossGradientOp final : public Operator<Context> {
 public:
    SoftmaxLossGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)),
          normalization(OperatorBase::GetSingleArg<string>("normalization", "VALID")) {
        vector<int> args = OperatorBase::GetRepeatedArg<int>("ignore_labels");
        if (args.size()) {
            ignore.Reshape(vector<TIndex>(1, args.size()));
            int* ignore_data = ignore.mutable_data<int, CPUContext>();
            for (int i = 0; i < args.size(); i++) ignore_data[i] = args[i];
        }
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim;
    Tensor ignore, valid;
    Tensor* prob;
    string normalization;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_LOSS_SOFTMAX_LOSS_OP_H_