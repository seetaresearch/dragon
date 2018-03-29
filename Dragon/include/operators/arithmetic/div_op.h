// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_DIV_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_DIV_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class DivOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(DivOp);
    USE_OPERATOR_FUNCTIONS(Context);

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
    USE_OPERATOR_FUNCTIONS(Context);

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

template <class Context>
class RDivOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RDivOp);
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

template <class Context>
class RDivGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RDivGradientOp);
    USE_OPERATOR_FUNCTIONS(Context);

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

}    // namepsace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_DIV_OP_H_