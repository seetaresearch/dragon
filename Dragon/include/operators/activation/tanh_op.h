// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_TANH_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_TANH_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class TanhOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(TanhOp);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class TanhGradientOp final : public Operator<Context> {
 public:
     TanhGradientOp(const OperatorDef& op_def, Workspace* ws)
         : Operator<Context>(op_def, ws) {
         DISABLE_SHARE_GRADIENT;
     }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_TANH_OP_H_