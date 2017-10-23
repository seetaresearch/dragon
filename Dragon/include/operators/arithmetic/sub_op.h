// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_SUB_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_SUB_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SubOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(SubOp);

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

template <class Context>
class SubGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(SubGradientOp);

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

template <class Context>
class RSubOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RSubOp);

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

template <class Context>
class RSubGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RSubGradientOp);

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_SUB_OP_H_