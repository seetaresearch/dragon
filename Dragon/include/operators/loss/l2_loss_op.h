// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_LOSS_L2_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_L2_LOSS_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class L2LossOp : public Operator<Context> {
 public:
    L2LossOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          normalization(OperatorBase::GetSingleArg<string>("normalization", "BATCH_SIZE")) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor* diff;
    string normalization;
};

template <class Context>
class L2LossGradientOp final : public Operator<Context> {
 public:
    L2LossGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          normalization(OperatorBase::GetSingleArg<string>("normalization", "BATCH_SIZE")) {}

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor* diff;
    string normalization;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_LOSS_L2_LOSS_OP_H_