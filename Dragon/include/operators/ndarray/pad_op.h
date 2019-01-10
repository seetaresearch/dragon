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

#ifndef DRAGON_OPERATORS_NDARRAY_PAD_OP_H_
#define DRAGON_OPERATORS_NDARRAY_PAD_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class PadOp final : public Operator<Context> {
 public:
    PadOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          pad_l(OperatorBase::Args<int64_t>("pad_l")),
          pad_r(OperatorBase::Args<int64_t>("pad_r")),
          mode(OperatorBase::Arg<string>("mode", "CONSTANT")),
          value(OperatorBase::Arg<float>("value", 0.f)) {
        if (pad_r.size() == 0) pad_r = pad_l;
        else CHECK_EQ(pad_l.size(), pad_r.size())
            << "The pad_l and pad_r should have the same length.";
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
    template <typename T> void ConstRunWithType();
    template <typename T> void ReflectRunWithType();
    template <typename T> void EdgeRunWithType();

 protected:
    float value;
    string mode;
    vector<int64_t> pad_l, pad_r, y_dimsV;
    Tensor l_padsT, x_dimsT, x_stridesT, y_dimsT;
};

template <class Context>
class PadGradientOp final : public Operator<Context> {
 public:
    PadGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          pad_l(OperatorBase::Args<int64_t>("pad_l")),
          pad_r(OperatorBase::Args<int64_t>("pad_r")),
          mode(OperatorBase::Arg<string>("mode", "CONSTANT")) {
        if (pad_r.size() == 0) pad_r = pad_l;
        else CHECK_EQ(pad_l.size(), pad_r.size())
            << "The pad_l and pad_r should have the same length.";
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
    template <typename T> void ConstRunWithType();
    template <typename T> void ReflectRunWithType();
    template <typename T> void EdgeRunWithType();

 protected:
    string mode;
    vector<int64_t> pad_l, pad_r, x_dimsV;
    Tensor l_padsT, x_dimsT, y_stridesT;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NDARRAY_PAD_OP_H_