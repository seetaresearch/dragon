// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_SIGMOID_OP_HPP
#define DRAGON_OPERATORS_ACTIVATION_SIGMOID_OP_HPP

#include "core/operator.h"

namespace dragon {

template <class Context>
class SigmoidOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(SigmoidOp);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class SigmoidGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(SigmoidGradientOp);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_SIGMOID_OP_HPP