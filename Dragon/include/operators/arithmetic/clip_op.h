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

#ifndef DRAGON_OPERATORS_ARITHMETIC_CLIP_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_CLIP_OP_H_

#include <float.h>
#include "core/operator.h"

namespace dragon {

template <class Context>
class ClipOp final : public Operator<Context> {
 public:
    ClipOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          low(OperatorBase::GetSingleArg<float>("low", -FLT_MAX)),
          high(OperatorBase::GetSingleArg<float>("high", FLT_MAX)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
    
 protected:
    float low, high;
};

template <class Context>
class ClipGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(ClipGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_CLIP_OP_H_