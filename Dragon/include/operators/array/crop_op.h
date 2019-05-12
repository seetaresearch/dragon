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

#ifndef DRAGON_OPERATORS_ARRAY_CROP_OP_H_
#define DRAGON_OPERATORS_ARRAY_CROP_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CropOp final : public Operator<Context> {
 public:
    CropOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          ofs_(OpArgs<int64_t>("offsets")),
          start_axis_(OpArg<int64_t>("start_axis", -1)),
          shape_desc_(OpArg<string>("shape_like", "")) {
        GET_ARGS_WITH_DESC(int64_t, starts);
        GET_ARGS_WITH_DESC(int64_t, sizes);
    }
    USE_OPERATOR_FUNCTIONS;

    void Setup();
    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    string shape_desc_;
    int64_t start_axis_;
    vec64_t st_, ed_, ofs_, keep_;
    Tensor X_starts_, X_strides_, Y_dims_;
    DECLARE_ARGS_WITH_DESC(int64_t, starts);
    DECLARE_ARGS_WITH_DESC(int64_t, sizes);
};

template <class Context>
class CropGradientOp final : public Operator<Context> {
 public:
    SIMPLE_CTOR_DTOR(CropGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    vec64_t st_, ed_;
    Tensor X_starts_, X_strides_, Y_dims_;
};

DEFINE_ARGS_WITH_DESC(int64_t, CropOp, starts);
DEFINE_ARGS_WITH_DESC(int64_t, CropOp, sizes);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_CROP_OP_H_