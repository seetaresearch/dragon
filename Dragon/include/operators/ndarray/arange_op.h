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
          start_desc(OperatorBase::GetSingleArg<string>("start", "")),
          stop_desc(OperatorBase::GetSingleArg<string>("stop", "")),
          step_desc(OperatorBase::GetSingleArg<string>("step", "")),
          dtype(OperatorBase::GetSingleArg<string>("dtype", "FLOAT32")) {}

    void Reshape();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    string start_desc, stop_desc, step_desc, dtype;
    TIndex start, stop, step, count;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_ARANGE_OP_H_