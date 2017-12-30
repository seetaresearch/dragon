// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_NDARRAY_RESHAPE_OP_H_
#define DRAGON_OPERATORS_NDARRAY_RESHAPE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ReshapeOp final : public Operator<Context> {
 public:
    ReshapeOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          shape(OperatorBase::GetRepeatedArg<int>("shape")) {
          new_shape.resize(shape.size());
    }

    void RunOnDevice() override;

 protected:
    vector<int> shape;
    vector<TIndex> new_shape;
};

template <class Context>
class ReshapeGradientOp final : public Operator<Context> {
 public:
    ReshapeGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {
        DISABLE_SHARE_GRADIENT;
    }

    void RunOnDevice() override;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_RESHAPE_OP_H_