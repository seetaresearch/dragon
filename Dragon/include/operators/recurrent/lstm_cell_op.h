// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_RECURRENT_LSTM_CELL_OP_H_
#define DRAGON_OPERATORS_RECURRENT_LSTM_CELL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class LSTMCellOp : public Operator<Context> {
 public:
    LSTMCellOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void ResetDesc();
    template <typename T> void RunWithType();
};

template <class Context>
class LSTMCellGradientOp : public Operator<Context> {
 public:
    LSTMCellGradientOp(const OperatorDef& op_def, Workspace* ws)
         : Operator<Context>(op_def, ws) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_RECURRENT_LSTM_CELL_OP_H_