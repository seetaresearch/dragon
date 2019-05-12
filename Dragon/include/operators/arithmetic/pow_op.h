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

#ifndef DRAGON_OPERATORS_ARITHMETIC_POW_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_POW_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class PowOp final : public Operator<Context> {
 public:
    PowOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          scale_(OpArg<float>("scale", 1.f)),
          shift_(OpArg<float>("shift", 0.f)),
          power_(OpArg<float>("power", 1.f)) {
        power_scale_ = power_ * scale_;
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float scale_, shift_;
    float power_, power_scale_;
};

template <class Context>
class PowGradientOp final : public Operator<Context> {
 public:
    PowGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          scale_(OpArg<float>("scale", 1.f)),
          shift_(OpArg<float>("shift", 0.f)),
          power_(OpArg<float>("power", 1.f)) {
        power_scale_ = power_ * scale_;
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float scale_, shift_;
    float power_, power_scale_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_POW_OP_H_