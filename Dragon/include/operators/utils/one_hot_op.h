// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UTILS_ONE_HOT_OP_H_
#define DRAGON_OPERATORS_UTILS_ONE_HOT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class OneHotOp final : public Operator < Context > {
 public:
    OneHotOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          depth(OperatorBase::GetSingleArg<int>("depth", -1)),
          on_value(OperatorBase::GetSingleArg<int>("on_value", 1)),
          off_value(OperatorBase::GetSingleArg<int>("off_value", 0)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex depth, on_value, off_value;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_UTILS_ONE_HOT_OP_H_