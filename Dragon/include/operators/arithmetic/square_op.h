// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_SQUARE_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_SQUARE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SquareOp final : public Operator<Context> {
public:
    USE_SIMPLE_CTOR_DTOR(SquareOp);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class SquareGradientOp final : public Operator<Context> {
public:
    USE_SIMPLE_CTOR_DTOR(SquareGradientOp);

    void ShareBeforeRun() override;
    void RunOnDevice() override;
    void ClearAfterRun() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_SQUARE_OP_H_