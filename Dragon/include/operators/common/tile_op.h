// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_COMMON_TILE_OP_H_
#define DRAGON_OPERATORS_COMMON_TILE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class TileOp : public Operator<Context> {
 public:
    TileOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws), 
          multiples(OperatorBase::GetRepeatedArg<int>("multiples")) {
        for (int i = 0; i < multiples.size(); i++)
            if (multiples[i] > 1)
                process_axes.push_back({ i, multiples[i] });
    }

    void RunOnDevice() override;
    template<typename T> void TileRunWithType();

 protected:
    vector<int> multiples;
    vector< pair<int, int> > process_axes;
    TIndex axis, multiple, outer_dim, dim, inner_dim;
    Tensor* dest, *source;
};

template <class Context>
class TileGradientOp : public Operator<Context> {
 public:
    TileGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          multiples(OperatorBase::GetRepeatedArg<int>("multiples")) {
        for (int i = (int)multiples.size() - 1; i >= 0; i--)
            if (multiples[i] > 1)
                process_axes.push_back({ i, multiples[i] });
    }

    void RunOnDevice() override;
    template<typename T> void TileRunWithType();

 protected:
    vector<int> multiples;
    vector< pair<int, int> > process_axes;
    TIndex axis, multiple, outer_dim, dim, inner_dim;
    Tensor* dest, *source;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_COMMON_TILE_OP_H_