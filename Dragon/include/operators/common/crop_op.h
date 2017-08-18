// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_COMMON_CROP_OP_H_
#define DRAGON_OPERATORS_COMMON_CROP_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CropOp: public Operator<Context> {
 public:
    CropOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 2)),
          offsets_param(OperatorBase::GetRepeatedArg<int>("offsets")),
          shape(OperatorBase::GetRepeatedArg<int>("shape")),
          shape_like(OperatorBase::GetSingleArg<string>("shape_like", "")) {
        CHECK(shape.size() * shape_like.size() == 0)
            << "\ncan not set shape and shape_like both.";
        CHECK(shape.size() + shape_like.size() != 0)
            << "\nmust set shape and shape_like either.";
    }

    void ComputeOutputShape();
    void RunOnDevice() override;
    template <typename T> void RunWithType();
    template <typename T> void RecursiveRunWithType(vector<TIndex> idxs, 
                                                    const vector<TIndex>& offsets,
                                                    int cur_dim,
                                                    Tensor* x,
                                                    Tensor* y);
 protected:
    TIndex axis;
    vector<int> offsets_param, shape;
    vector<TIndex> output_shape, offsets;
    string shape_like;
};

template <class Context>
class CropGradientOp final : public Operator<Context > {
 public:
    CropGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 2)),
          offsets_param(OperatorBase::GetRepeatedArg<int>("offsets")),
          shape(OperatorBase::GetRepeatedArg<int>("shape")),
          shape_like(OperatorBase::GetSingleArg<string>("shape_like", "")) {
        CHECK(shape.size() * shape_like.size() == 0)
            << "\ncan not set shape and shape_like both.";
        CHECK(shape.size() + shape_like.size() != 0)
            << "\nmust set shape and shape_like either.";
    }

    void ComputeOutputShape();
    void RunOnDevice() override;
    template <typename T> void RunWithType();
    template <typename T> void RecursiveRunWithType(vector<TIndex> idxs, 
                                                    const vector<TIndex>& offsets,
                                                    int cur_dim,
                                                    Tensor* dy,
                                                    Tensor* dx);
 protected:
    TIndex axis;
    vector<int> offsets_param, shape;
    vector<TIndex> output_shape, offsets;
    string shape_like;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_COMMON_CROP_OP_H_