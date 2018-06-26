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

#ifndef DRAGON_OPERATORS_UPDATE_ADAM_UPDATE_OP_H_
#define DRAGON_OPERATORS_UPDATE_ADAM_UPDATE_OP_H_

#include "operators/update/update_op_base.h"

namespace dragon {

template <class Context>
class AdamUpdateOp final : public UpdateOpBase<Context> {
 public:
    AdamUpdateOp(const OperatorDef& op_def, Workspace* ws) 
        : UpdateOpBase<Context>(op_def, ws), t(0) {}
    USE_OPERATOR_FUNCTIONS;
    USE_UPDATER_FUNCTIONS(Context);

    void ComputeRunWithFloat() override;
    void ComputeRunWithFloat16() override;

 protected:
    int t; float lr, beta1, beta2, eps;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_UPDATE_ADAM_UPDATE_OP_H_