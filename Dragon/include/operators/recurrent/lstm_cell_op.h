/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_RECURRENT_LSTM_CELL_OP_H_
#define DRAGON_OPERATORS_RECURRENT_LSTM_CELL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class LSTMCellOp final : public Operator<Context> {
 public:
    LSTMCellOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
};

template <class Context>
class LSTMCellGradientOp final : public Operator<Context> {
 public:
    LSTMCellGradientOp(const OperatorDef& def, Workspace* ws)
         : Operator<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_RECURRENT_LSTM_CELL_OP_H_