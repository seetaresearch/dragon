// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

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

    void ShareBeforeRun() override;
    void RunOnDevice() override;
    void ClearAfterRun() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, dim, inner_dim;
    TIndex x_offset, y_offset;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_GRAM_MATRIX_OP_H_