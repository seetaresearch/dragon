// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_COMMON_ARGMAX_OP_H_
#define DRAGON_OPERATORS_COMMON_ARGMAX_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ArgmaxOp final : public Operator<Context> {
 public:
    ArgmaxOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 0)),
          top_k(OperatorBase::GetSingleArg<int>("top_k", 1)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, top_k, count, inner_dim;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_COMMON_ARGMAX_OP_H_