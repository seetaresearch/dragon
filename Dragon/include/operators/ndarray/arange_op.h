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
          dtype(OperatorBase::GetSingleArg<string>("dtype", "FLOAT32")) {
        GET_ARGUMENT_WITH_DESC(int, start, 0);
        GET_ARGUMENT_WITH_DESC(int, stop, 0);
        GET_ARGUMENT_WITH_DESC(int, step, 1);
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    DECLARE_ARGUMENT_WITH_DESC(int, start);
    DECLARE_ARGUMENT_WITH_DESC(int, stop);
    DECLARE_ARGUMENT_WITH_DESC(int, step);
    string dtype;
};

DEFINE_ARGUMENT_WITH_DESC(int, ArangeOp, start);
DEFINE_ARGUMENT_WITH_DESC(int, ArangeOp, stop);
DEFINE_ARGUMENT_WITH_DESC(int, ArangeOp, step);

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_ARANGE_OP_H_