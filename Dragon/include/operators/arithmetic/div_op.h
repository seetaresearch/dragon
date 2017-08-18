// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_DIV_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_DIV_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class DivOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(DivOp);

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

template <class Context>
class DivGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(DivGradientOp);

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

}    // namepsace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_DIV_OP_H_