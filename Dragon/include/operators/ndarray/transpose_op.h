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
    TransposeOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          perms(OperatorBase::GetRepeatedArg<int>("perms")) {
        if (perms.size() > 0) reverse_dims = false;
        else reverse_dims = true;
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    vector<int> perms;
    bool reverse_dims;
    Tensor* order, *old_steps, *new_steps;
};


template <class Context>
class TransposeGradientOp final : public Operator<Context> {
 public:
    TransposeGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor* order, *old_steps, *new_steps;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_TRANSPOSE_OP_H_