// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_NDARRAY_CROP_OP_H_
#define DRAGON_OPERATORS_NDARRAY_CROP_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CropOp: public Operator<Context> {
 public:
    CropOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          starts(OperatorBase::GetRepeatedArg<int>("starts")),
          ends(OperatorBase::GetRepeatedArg<int>("ends")),
          start_axis(OperatorBase::GetSingleArg<int>("start_axis", -1)),
          offsets(OperatorBase::GetRepeatedArg<int>("offsets")),
          shape(OperatorBase::GetRepeatedArg<int>("shape")),
          shape_like(OperatorBase::GetSingleArg<string>("shape_like", "")) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex start_axis;
    string shape_like;
    vector<int> starts, ends, offsets, shape;
    vector< pair<int, int> > process_axes;
    TIndex axis, inner_dim, dim;
    Tensor* dest, *source;
};

template <class Context>
class CropGradientOp final : public Operator<Context > {
 public:
    CropGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          starts(OperatorBase::GetRepeatedArg<int>("starts")),
          ends(OperatorBase::GetRepeatedArg<int>("ends")),
          start_axis(OperatorBase::GetSingleArg<int>("start_axis", -1)),
          offsets(OperatorBase::GetRepeatedArg<int>("offsets")),
          shape(OperatorBase::GetRepeatedArg<int>("shape")),
          shape_like(OperatorBase::GetSingleArg<string>("shape_like", "")) {
        DISABLE_SHARE_GRADIENT;
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex start_axis;
    string shape_like;
    vector<int> starts, ends, offsets, shape;
    vector< pair<int, int> > process_axes;
    TIndex axis, inner_dim, dim;
    Tensor* dest, *source;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_CROP_OP_H_