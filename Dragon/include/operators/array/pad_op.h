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

#ifndef DRAGON_OPERATORS_ARRAY_PAD_OP_H_
#define DRAGON_OPERATORS_ARRAY_PAD_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class PadOp final : public Operator<Context> {
 public:
    PadOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          pad_l_(OpArgs<int64_t>("pad_l")),
          pad_r_(OpArgs<int64_t>("pad_r")),
          mode_(OpArg<string>("mode", "CONSTANT")),
          value_(OpArg<float>("value", 0.f)) {
        if (pad_r_.empty()) {
            pad_r_ = pad_l_;
        } else {
            CHECK_EQ(pad_l_.size(), pad_r_.size())
                << "\nThe pad_l and pad_r "
                << "should have the same length.";
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
    template <typename T> void ConstRunImpl();
    template <typename T> void ReflectRunImpl();
    template <typename T> void EdgeRunImpl();

 protected:
    float value_;
    string mode_;
    vec64_t pad_l_, pad_r_;
    Tensor pads_, X_dims_, X_strides_, Y_dims_;
};

template <class Context>
class PadGradientOp final : public Operator<Context> {
 public:
    PadGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          pad_l_(OpArgs<int64_t>("pad_l")),
          pad_r_(OpArgs<int64_t>("pad_r")),
          mode_(OpArg<string>("mode", "CONSTANT")) {
        if (pad_r_.empty()) {
            pad_r_ = pad_l_;
        } else {
            CHECK_EQ(pad_l_.size(), pad_r_.size())
                << "\nThe pad_l and pad_r "
                << "should have the same length.";
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
    template <typename T> void ConstRunImpl();
    template <typename T> void ReflectRunImpl();
    template <typename T> void EdgeRunImpl();

 protected:
    string mode_;
    vec64_t pad_l_, pad_r_;
    Tensor pads_, X_dims_, Y_strides_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_PAD_OP_H_