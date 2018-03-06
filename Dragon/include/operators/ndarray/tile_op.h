// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_NDARRAY_TILE_OP_H_
#define DRAGON_OPERATORS_NDARRAY_TILE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class TileOp : public Operator<Context> {
 public:
    TileOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {
        GET_ARGUMENTS_WITH_DESC(int, multiples);
    }

    void RunOnDevice() override;
    template<typename T> void TileRunWithType();

 protected:
    DECLARE_ARGUMENTS_WITH_DESC(int, multiples);
    TIndex axis, multiple, outer_dim, ex_inner_dim;
    Tensor* dest, *source;
};

template <class Context>
class TileGradientOp : public Operator<Context> {
 public:
    TileGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws) {
        GET_ARGUMENTS_WITH_DESC(int, multiples);
        DISABLE_SHARE_GRADIENT;
    }

    void RunOnDevice() override;
    template<typename T> void TileRunWithType();

 protected:
    DECLARE_ARGUMENTS_WITH_DESC(int, multiples);
    TIndex axis, multiple, outer_dim, ex_inner_dim;
    Tensor* dest, *source;
};

DEFINE_ARGUMENTS_WITH_DESC(int, TileOp, multiples);
DEFINE_ARGUMENTS_WITH_DESC(int, TileGradientOp, multiples);

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_TILE_OP_H_