/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_NDARRAY_CROP_OP_H_
#define DRAGON_OPERATORS_NDARRAY_CROP_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CropOp final : public Operator<Context> {
 public:
    CropOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          start_axis(OperatorBase::Arg<int64_t>("start_axis", -1)),
          offsets(OperatorBase::Args<int64_t>("offsets")),
          shape_like(OperatorBase::Arg<string>("shape_like", "")) {
        GET_ARGUMENTS_WITH_DESC(int64_t, starts);
        GET_ARGUMENTS_WITH_DESC(int64_t, sizes);
    }
    USE_OPERATOR_FUNCTIONS;

    void Setup();
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int64_t start_axis;
    string shape_like;
    vector<int64_t> offsets;
    vector<int64_t> st, ed, keep_dims, y_dimsV;
    Tensor startsT, x_stridesT, y_dimsT;
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, starts);
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, sizes);
};

DEFINE_ARGUMENTS_WITH_DESC(int64_t, CropOp, starts);
DEFINE_ARGUMENTS_WITH_DESC(int64_t, CropOp, sizes);

template <class Context>
class CropGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(CropGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    vector<int64_t> st, ed, y_dimsV;
    Tensor startsT, x_stridesT, y_dimsT;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NDARRAY_CROP_OP_H_