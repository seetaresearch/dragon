// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_RECURRENT_LSTM_UNIT_OP_H_
#define DRAGON_OPERATORS_RECURRENT_LSTM_UNIT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class LSTMUnitOp : public Operator<Context> {
 public:
    LSTMUnitOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          has_cont(OperatorBase::GetSingleArg<string>("cont_t", "")) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex num, channels;
    string has_cont;
    Tensor* cont_t;
};

template <class Context>
class LSTMUnitGradientOp : public Operator<Context> {
 public:
    LSTMUnitGradientOp(const OperatorDef& op_def, Workspace* ws)
         : Operator<Context>(op_def, ws) {
         this->allow_share_grads_ = false;
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex num, channels;
    Tensor* zeros;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_RECURRENT_LSTM_UNIT_OP_H_