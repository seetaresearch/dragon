// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_OPERATORS_MISC_ASTYPE_OP_H_
#define DRAGON_OPERATORS_MISC_ASTYPE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class AsTypeOp final : public Operator<Context> {
 public:
    AsTypeOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
        dtype(OperatorBase::GetSingleArg<string>("dtype", "float32")),
        inplace(OperatorBase::GetSingleArg<bool>("inplace", false)) {}
     USE_OPERATOR_FUNCTIONS;

     void RunOnDevice() override;

 protected:
    string dtype;
    bool inplace;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_MISC_ASTYPE_OP_H_