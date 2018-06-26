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

#ifndef DRAGON_OPERATORS_NDARRAY_FLATTEN_OP_H_
#define DRAGON_OPERATORS_NDARRAY_FLATTEN_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class FlattenOp final : public Operator<Context> {
 public:
    FlattenOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 0)),
          num_axes(OperatorBase::GetSingleArg<int>("num_axes", -1)),
          keep_axes(OperatorBase::GetSingleArg<int>("keep_axes", INT_MAX)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    void SqueezeRun();
    void KeepRun();

 protected:
    TIndex axis, num_axes, keep_axes;
};

template <class Context>
class FlattenGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(FlattenGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_FLATTEN_OP_H_