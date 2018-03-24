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
          shape_like_desc(OperatorBase::GetSingleArg<string>("shape_like", "")) {
        GET_ARGUMENTS_WITH_DESC(int, shape);
    }

    void RunOnDevice() override;

 protected:
    DECLARE_ARGUMENTS_WITH_DESC(int, shape);
    string shape_like_desc;
    vector<TIndex> require_shape, new_shape;
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

DEFINE_ARGUMENTS_WITH_DESC(int, ReshapeOp, shape);

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_RESHAPE_OP_H_