// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_NDARRAY_REDUCE_OP_H_
#define DRAGON_OPERATORS_NDARRAY_REDUCE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ReduceOp final : public Operator<Context> {
 public:
    ReduceOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          operation(OperatorBase::GetSingleArg<string>("operation", "NONE")),
          keep_dims(OperatorBase::GetSingleArg<bool>("keep_dims", false)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void SumRunWithType();
    template <typename T> void MeanRunWithType();

 protected:
    bool keep_dims;
    string operation;
    TIndex axis, axis_dim, count, inner_dim;
    Tensor* multiplier;
};

template <class Context>
class ReduceGradientOp final : public Operator<Context> {
 public:
    ReduceGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
        axis(OperatorBase::GetSingleArg<int>("axis", -1)),
        operation(OperatorBase::GetSingleArg<string>("operation", "NONE")) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void SumRunWithType();
    template <typename T> void MeanRunWithType();

 protected:
    string operation;
    TIndex axis, axis_dim, count, inner_dim;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_REDUCE_OP_H_