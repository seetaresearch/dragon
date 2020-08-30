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

#ifndef DRAGON_OPERATORS_ARRAY_PAD_OP_H_
#define DRAGON_OPERATORS_ARRAY_PAD_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class PadOp final : public Operator<Context> {
 public:
  PadOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        value_(OP_SINGLE_ARG(float, "value", 0.f)),
        mode_(OP_SINGLE_ARG(string, "mode", "CONSTANT")) {
    INIT_OP_REPEATED_ARG_WITH_DESC(int64_t, pads);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float value_;
  string mode_;
  DECLARE_OP_REPEATED_ARG_WITH_DESC(int64_t, pads);
};

template <class Context>
class PadGradientOp final : public Operator<Context> {
 public:
  PadGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        pad_l_(OP_REPEATED_ARG(int64_t, "pad_l")),
        pad_r_(OP_REPEATED_ARG(int64_t, "pad_r")),
        mode_(OP_SINGLE_ARG(string, "mode", "CONSTANT")) {
    if (pad_r_.empty()) {
      pad_r_ = pad_l_;
    } else {
      CHECK_EQ(pad_l_.size(), pad_r_.size())
          << "\nThe <pad_l> and <pad_r> should have the same length.";
    }
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  string mode_;
  vec64_t pad_l_, pad_r_;
};

DEFINE_OP_REPEATED_ARG_WITH_DESC(int64_t, PadOp, pads);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_PAD_OP_H_
