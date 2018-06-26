// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_ADD_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_ADD_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class AddOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(AddOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class AddGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(AddGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class RAddOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RAddOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class RAddGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RAddGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_ADD_OP_H_