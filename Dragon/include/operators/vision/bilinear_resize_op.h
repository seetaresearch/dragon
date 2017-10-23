// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_BILINEAR_RESIZE_OP_H_
#define DRAGON_OPERATORS_VISION_BILINEAR_RESIZE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class BilinearResizeOp : public Operator<Context> {
 public:
    BilinearResizeOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          static_dsize(OperatorBase::GetRepeatedArg<int>("static_dsize")),
          dynamic_dsize(OperatorBase::GetRepeatedArg<string>("dynamic_dsize")),
          fy(OperatorBase::GetSingleArg<float>("fy", -1.0)),
          fx(OperatorBase::GetSingleArg<float>("fx", -1.0)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    vector<int> static_dsize;
    vector<string> dynamic_dsize;
    vector<TIndex> dims;
    float h_scale, w_scale, fy, fx;
};

template <class Context>
class BilinearResizeGradientOp : public Operator<Context> {
 public:
    BilinearResizeGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_BILINEAR_RESIZE_OP_H_