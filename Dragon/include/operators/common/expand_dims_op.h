// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_COMMON_EXPAND_DIMS_OP_H_
#define DRAGON_OPERATORS_COMMON_EXPAND_DIMS_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ExpandDimsOp final : public Operator<Context> {
 public:
    ExpandDimsOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)) {}

    void RunOnDevice() override;

 protected:
    TIndex axis;
};

template <class Context>
class ExpandDimsGradientOp final : public Operator<Context> {
 public:
     ExpandDimsGradientOp(const OperatorDef& op_def, Workspace* ws)
         : Operator<Context>(op_def, ws) {
         DISABLE_SHARE_GRADIENT;
     }
    void RunOnDevice() override;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_COMMON_EXPAND_DIMS_OP_H_
