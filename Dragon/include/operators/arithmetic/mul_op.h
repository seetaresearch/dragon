// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_MUL_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_MUL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class MulOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(MulOp);

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

template <class Context>
class MulGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(MulGradientOp);

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_MUL_OP_H_