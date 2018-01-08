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
        : Operator<Context>(op_def, ws), 
          multiples_desc(OperatorBase::GetRepeatedArg<string>("multiples")) {}

    void RunOnDevice() override;
    template<typename T> void TileRunWithType();

 protected:
    vector<string> multiples_desc;
    TIndex axis, multiple, outer_dim, ex_inner_dim;
    Tensor* dest, *source;
};

template <class Context>
class TileGradientOp : public Operator<Context> {
 public:
    TileGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
        multiples_desc(OperatorBase::GetRepeatedArg<string>("multiples")) {
        DISABLE_SHARE_GRADIENT;
    }

    void RunOnDevice() override;
    template<typename T> void TileRunWithType();

 protected:
    vector<string> multiples_desc;
    TIndex axis, multiple, outer_dim, ex_inner_dim;
    Tensor* dest, *source;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_TILE_OP_H_