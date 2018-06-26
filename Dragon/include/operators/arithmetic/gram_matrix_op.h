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

#ifndef DRAGON_OPERATORS_ARITHMETIC_GRAM_MATRIX_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_GRAM_MATRIX_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class GramMatrixOp final : public Operator<Context> {
 public:
    GramMatrixOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, dim, inner_dim;
    TIndex x_offset, y_offset;
};

template <class Context>
class GramMatrixGradientOp final : public Operator<Context> {
 public:
    GramMatrixGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, dim, inner_dim;
    TIndex x_offset, y_offset;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_GRAM_MATRIX_OP_H_