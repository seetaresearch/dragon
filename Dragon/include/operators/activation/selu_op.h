// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_SELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_SELU_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SEluOp : public Operator<Context> {
 public:
    SEluOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class SEluGradientOp : public Operator<Context> {
 public:
    SEluGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {
        DISABLE_SHARE_GRADIENT;
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_SELU_OP_H_