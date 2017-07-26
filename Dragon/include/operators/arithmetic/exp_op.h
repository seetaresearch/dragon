// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_EXP_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_EXP_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ExpOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(ExpOp);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class ExpGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(ExpGradientOp);

    void ShareBeforeRun() override;
    void RunOnDevice() override;
    void ClearAfterRun() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_EXP_OP_H_