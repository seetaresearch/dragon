// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UTILS_COMPARE_OP_H_
#define DRAGON_OPERATORS_UTILS_COMPARE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CompareOp final : public Operator<Context> {
 public:
    CompareOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          operation(OperatorBase::GetSingleArg<string>("operation", "NONE")) {}

    void RunOnDevice() override;
    template <typename T> void EqualRunWithType();

 protected:
    string operation;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_UTILS_COMPARE_OP_H_