/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_RECURRENT_RNN_PARAM_OP_H_
#define DRAGON_OPERATORS_RECURRENT_RNN_PARAM_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class RNNParamSetOp final : public Operator<Context> {
 public:
  RNNParamSetOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        param_type_(OP_SINGLE_ARG(string, "param_type", "matrix")),
        num_layers_(OP_SINGLE_ARG(int64_t, "num_layers", 1)),
        num_directions_(OP_SINGLE_ARG(int64_t, "bidirectional", 0) + 1),
        input_size_(OP_SINGLE_ARG(int64_t, "input_size", 0)),
        hidden_size_(OP_SINGLE_ARG(int64_t, "hidden_size", 0)),
        layer_id_(OP_SINGLE_ARG(int64_t, "layer_id", 0)),
        param_id_(OP_SINGLE_ARG(int64_t, "param_id", 0)) {
    auto mode_str = OP_SINGLE_ARG(string, "rnn_mode", "rnn_tanh");
    if (mode_str == "rnn_tanh") {
      num_params_ = 2;
      spliter_ = 1;
    } else if (mode_str == "rnn_relu") {
      num_params_ = 2;
      spliter_ = 1;
    } else if (mode_str == "lstm") {
      num_params_ = 8;
      spliter_ = 4;
    } else if (mode_str == "gru") {
      num_params_ = 6;
      spliter_ = 3;
    } else {
      LOG(FATAL) << "Unknown RNN Mode: " << mode_str;
    }
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  string param_type_;
  int64_t layer_id_, param_id_;
  int64_t num_layers_, num_directions_;
  int64_t num_params_, spliter_;
  int64_t input_size_, hidden_size_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_RECURRENT_RNN_PARAM_OP_H_
