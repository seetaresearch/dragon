// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_

#include "core/operator.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
class DropoutOp final : public Operator<Context> {
 public:
    DropoutOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          prob(OperatorBase::GetSingleArg<float>("prob", 0.5)) {
        bool use_scale = OperatorBase::GetSingleArg<bool>("scale", true);
        threshold = static_cast<unsigned int>(UINT_MAX * prob);
        if (use_scale) scale = 1.0 / (1.0 - prob);
        else scale = 1.0;
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float prob, scale;
    unsigned int threshold;
    Tensor* mask;
};

template <class Context>
class DropoutGradientOp final : public Operator<Context> {
 public:
    DropoutGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          prob(OperatorBase::GetSingleArg<float>("prob", 0.5)) {
        bool use_scale = OperatorBase::GetSingleArg<bool>("scale", true);
        threshold = static_cast<unsigned int>(UINT_MAX * prob);
        if (use_scale) scale = 1.0 / (1.0 - prob);
        else scale = 1.0;
        DISABLE_SHARE_GRADIENT;
    }

    void RunOnDevice() override;
    void CleanResource() override;
    template <typename T> void RunWithType();

 protected:
     float prob, scale;
     unsigned int threshold;
     Tensor* mask;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_