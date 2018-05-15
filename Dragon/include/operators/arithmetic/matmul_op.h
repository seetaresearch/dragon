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

#ifndef DRAGON_OPERATORS_ARITHMETIC_MATMUL_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_MATMUL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class MatmulOp final : public Operator<Context> {
 public:
    MatmulOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          transA(OperatorBase::GetSingleArg<bool>("TransA", false)),
          transB(OperatorBase::GetSingleArg<bool>("TransB", false)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    bool transA, transB;
    TIndex n, x1_offset, x2_offset, y_offset;
    TIndex M, K1, K2, N;
};

template <class Context>
class MatmulGradientOp final : public Operator<Context> {
 public:
    MatmulGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
        transA(OperatorBase::GetSingleArg<bool>("TransA", false)),
        transB(OperatorBase::GetSingleArg<bool>("TransB", false)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    bool transA, transB;
    TIndex n, x1_offset, x2_offset, y_offset;
    TIndex M, K1, K2, N;
};
    
}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_MATMUL_OP_H_