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

#ifndef DRAGON_OPERATORS_ARITHMETIC_MUL_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_MUL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class MulOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(MulOp);
    USE_OPERATOR_FUNCTIONS(Context);

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
    USE_OPERATOR_FUNCTIONS(Context);

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

template <class Context>
class RMulOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RMulOp);
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

template <class Context>
class RMulGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RMulGradientOp);
    USE_OPERATOR_FUNCTIONS(Context);

    void ShareGradient() override;
    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor* bcast_multiplier;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_MUL_OP_H_