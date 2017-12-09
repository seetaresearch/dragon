// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

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

    void RunOnDevice() override;
    void SqueezeRun();
    void KeepRun();

 protected:
    TIndex axis, num_axes, keep_axes;
};

template <class Context>
class FlattenGradientOp final : public Operator<Context> {
 public:
    FlattenGradientOp(const OperatorDef& op_def, Workspace* ws)
         : Operator<Context>(op_def, ws) {}

    void RunOnDevice() override;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_FLATTEN_OP_H_