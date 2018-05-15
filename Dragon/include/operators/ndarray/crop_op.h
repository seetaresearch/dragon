// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
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
          start_axis(OperatorBase::GetSingleArg<int>("start_axis", -1)),
          offsets(OperatorBase::GetRepeatedArg<int>("offsets")),
          shape(OperatorBase::GetRepeatedArg<int>("shape")),
          shape_like(OperatorBase::GetSingleArg<string>("shape_like", "")) {
        GET_ARGUMENTS_WITH_DESC(int, starts);
        GET_ARGUMENTS_WITH_DESC(int, ends);
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex start_axis;
    string shape_like;
    vector<int> st, ed, offsets, shape, keep_dims;
    DECLARE_ARGUMENTS_WITH_DESC(int, starts);
    DECLARE_ARGUMENTS_WITH_DESC(int, ends);
    vector< pair<int, int> > process_axes;
    TIndex axis, inner_dim, dim;
    Tensor* dest, *source;
};

DEFINE_ARGUMENTS_WITH_DESC(int, CropOp, starts);
DEFINE_ARGUMENTS_WITH_DESC(int, CropOp, ends);

template <class Context>
class CropGradientOp final : public Operator<Context > {
 public:
    USE_SIMPLE_CTOR_DTOR(CropGradientOp);
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    vector<int> st, ed, offsets, keep_dims;
    vector< pair<int, int> > process_axes;
    TIndex axis, inner_dim, dim;
    Tensor* dest, *source;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_CROP_OP_H_