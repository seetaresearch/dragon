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

#ifndef DRAGON_OPERATORS_NDARRAY_TRANSPOSE_OP_H_
#define DRAGON_OPERATORS_NDARRAY_TRANSPOSE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class TransposeOp final: public Operator<Context> {
 public:
    TransposeOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {
        GET_ARGUMENTS_WITH_DESC(int, perms);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    DECLARE_ARGUMENTS_WITH_DESC(int, perms);
    Tensor* order, *old_steps, *new_steps;
};

DEFINE_ARGUMENTS_WITH_DESC(int, TransposeOp, perms);

template <class Context>
class TransposeGradientOp final : public Operator<Context> {
 public:
    TransposeGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor* order, *old_steps, *new_steps;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_TRANSPOSE_OP_H_