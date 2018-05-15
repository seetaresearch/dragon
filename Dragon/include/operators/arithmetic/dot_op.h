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

#ifndef DRAGON_OPERATORS_ARITHMETIC_DOT_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_DOT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class DotOp final : public Operator<Context> {
 public:
    DotOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          transA(OperatorBase::GetSingleArg<bool>("TransA", false)),
          transB(OperatorBase::GetSingleArg<bool>("TransB", false)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void DotRunWithType();
    template <typename T> void GemmRunWithType();
    template <typename T> void GemvRunWithType();

 protected:
    bool transA, transB;
    TIndex M, K1, K2, N1, N2;
};

template <class Context>
class DotGradientOp final : public Operator<Context> {
 public:
    DotGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
        transA(OperatorBase::GetSingleArg<bool>("TransA", false)),
        transB(OperatorBase::GetSingleArg<bool>("TransB", false)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void DotRunWithType();
    template <typename T> void GemmRunWithType();
    template <typename T> void GemvRunWithType();

 protected:
    bool transA, transB;
    TIndex M, K1, K2, N1, N2;
};
    
}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_DOT_OP_H_