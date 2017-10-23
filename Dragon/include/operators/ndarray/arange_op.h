// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_NDARRAY_ARGMAX_OP_H_
#define DRAGON_OPERATORS_NDARRAY_ARGMAX_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ArangeOp final : public Operator<Context> {
 public:
    ArangeOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          start(OperatorBase::GetSingleArg<int>("static_start", 0)),
          stop(OperatorBase::GetSingleArg<int>("static_stop", -1)),
          step(OperatorBase::GetSingleArg<int>("static_step", 1)),
          dtype(OperatorBase::GetSingleArg<string>("dtype", "FLOAT32")) {
        dynamic_start_ = OperatorBase::GetSingleArg<string>("dynamic_start", "");
        dynamic_stop_ = OperatorBase::GetSingleArg<string>("dynamic_stop", "");
        dynamic_step_ = OperatorBase::GetSingleArg<string>("dynamic_step", "");
    }

    void RunOnDevice() override;
    void Reshape();
    template <typename T> void RunWithType();

 protected:
    TIndex start, stop, step, count;
    Tensor* dynamic_start, *dynamic_stop, *dynamic_step;
    string dynamic_start_, dynamic_stop_, dynamic_step_;
    string dtype;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_ARANGE_OP_H_