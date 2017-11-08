// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_PRELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_PRELU_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class PReluOp : public Operator<Context> {
 public:
    PReluOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          channel_shared(OperatorBase::GetSingleArg<bool>("channel_shared", false)),
          data_format(OperatorBase::GetSingleArg<string>("data_format", "NCHW")) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    bool channel_shared;
    string data_format;
    TIndex channels, dim;
};

template <class Context>
class PReluGradientOp : public Operator<Context> {
 public:
    PReluGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          channel_shared(OperatorBase::GetSingleArg<bool>("channel_shared", false)),
          data_format(OperatorBase::GetSingleArg<string>("data_format", "NCHW")) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    bool channel_shared;
    string data_format;
    TIndex channels, dim;
    Tensor* bcast_dw, *multiplier;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_PRELU_OP_H_