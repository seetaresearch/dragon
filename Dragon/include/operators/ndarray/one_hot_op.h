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

#ifndef DRAGON_OPERATORS_NDARRAY_ONE_HOT_OP_H_
#define DRAGON_OPERATORS_NDARRAY_ONE_HOT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class OneHotOp final : public Operator < Context > {
 public:
    OneHotOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          depth(OperatorBase::GetSingleArg<int>("depth", -1)),
          on_value(OperatorBase::GetSingleArg<int>("on_value", 1)),
          off_value(OperatorBase::GetSingleArg<int>("off_value", 0)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex depth, on_value, off_value;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_ONE_HOT_OP_H_