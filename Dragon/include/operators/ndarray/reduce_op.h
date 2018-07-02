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
    ReduceOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", -1)),
          operation(OperatorBase::Arg<string>("operation", "NONE")),
          keep_dims(OperatorBase::Arg<bool>("keep_dims", false)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void SumRunWithType();
    template <typename T> void MeanRunWithType();

 protected:
    TIndex axis, keep_dims, axis_dim, count, inner_dim;
    string operation;
};

template <class Context>
class ReduceGradientOp final : public Operator<Context> {
 public:
    ReduceGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", -1)),
          operation(OperatorBase::Arg<string>("operation", "NONE")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void SumRunWithType();
    template <typename T> void MeanRunWithType();

 protected:
    TIndex axis, axis_dim, count, inner_dim;
    string operation;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_REDUCE_OP_H_