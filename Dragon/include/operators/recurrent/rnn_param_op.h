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

#ifndef DRAGON_OPERATORS_RECURRENT_RNN_PARAM_OP_H_
#define DRAGON_OPERATORS_RECURRENT_RNN_PARAM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class RNNParamSetOp final : public Operator<Context> {
 public:
    RNNParamSetOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          param_type(OperatorBase::Arg<string>("param_type", "matrix")),
          rnn_mode(OperatorBase::Arg<string>("rnn_mode", "rnn_tanh")),
          num_layers(OperatorBase::Arg<int>("num_layers", 1)),
          num_directions(OperatorBase::Arg<int>("num_directions", 1)),
          input_size(OperatorBase::Arg<int>("input_size", 0)),
          hidden_size(OperatorBase::Arg<int>("hidden_size", 0)),
          layer_id(OperatorBase::Arg<int>("layer_id", 0)),
          param_id(OperatorBase::Arg<int>("param_id", 0)) {
        if (rnn_mode == "rnn_tanh") { num_params = 2; spliter = 1; }
        else if (rnn_mode == "rnn_relu") { num_params = 2; spliter = 1; }
        else if (rnn_mode == "lstm") { num_params = 8; spliter = 4; }
        else if (rnn_mode == "gru") { num_params = 6; spliter = 3; }
        else LOG(FATAL) << "Unsupported rnn mode: " << rnn_mode;
        input_ex_size = hidden_size * num_directions;
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    string param_type, rnn_mode;
    TIndex num_layers, num_directions, num_params, spliter;
    TIndex input_size, input_ex_size, hidden_size;
    TIndex layer_id, param_id;
};

}    // namespace dragon

#endif