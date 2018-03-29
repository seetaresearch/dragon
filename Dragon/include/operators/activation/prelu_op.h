// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

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
    USE_OPERATOR_FUNCTIONS(Context);

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
    USE_OPERATOR_FUNCTIONS(Context);

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