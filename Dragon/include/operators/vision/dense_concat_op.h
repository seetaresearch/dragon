// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_DENSE_CONCAT_OP_H_
#define DRAGON_OPERATORS_VISION_DENSE_CONCAT_OP_H_

#include "operators/common/concat_op.h"

namespace dragon {

template <class Context>
class DenseConcatOp final : public ConcatOp<Context> {
 public:
     DenseConcatOp(const OperatorDef& op_def, Workspace* ws)
         : ConcatOp<Context>(op_def, ws) {}
};

template <class Context>
class DenseConcatGradientOp : public ConcatGradientOp<Context> {
 public:
    DenseConcatGradientOp(const OperatorDef& op_def, Workspace* ws)
        : ConcatGradientOp<Context>(op_def, ws),
          growth_rate(OperatorBase::GetSingleArg<int>("growth_rate", 0)) {}

    void ElimateCorruption() override;
    template <typename T> void RestoreX1();

 protected:
     TIndex growth_rate;
};


}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_DENSE_CONCAT_OP_H_