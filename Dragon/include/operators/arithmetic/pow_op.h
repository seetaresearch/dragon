// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_POW_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_POW_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class PowOp: public Operator<Context> {
 public:
    PowOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          scale(OperatorBase::GetSingleArg<float>("scale", 1.0)),
          shift(OperatorBase::GetSingleArg<float>("shift", 0.0)),
          power(OperatorBase::GetSingleArg<float>("power", 1.0)) {
          power_scale = power * scale;
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float scale, shift, power, power_scale;
};

template <class Context>
class PowGradientOp final : public Operator<Context> {
 public:
    PowGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
        scale(OperatorBase::GetSingleArg<float>("scale", 1.0)),
        shift(OperatorBase::GetSingleArg<float>("shift", 0.0)),
        power(OperatorBase::GetSingleArg<float>("power", 1.0)) {
        power_scale = power * scale;
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float scale, shift, power, power_scale;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_POW_OP_H_