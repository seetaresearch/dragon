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

#ifndef DRAGON_OPERATORS_UPDATE_RMSPROP_UPDATE_OP_H_
#define DRAGON_OPERATORS_UPDATE_RMSPROP_UPDATE_OP_H_

#include "operators/update/update_op_base.h"

namespace dragon {

template <class Context>
class RMSPropUpdateOp final : public UpdateOpBase<Context> {
 public:
    RMSPropUpdateOp(const OperatorDef& def, Workspace* ws)
        : UpdateOpBase<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;
    USE_UPDATER_FUNCTIONS(Context);

    void ComputeRunWithFloat32() override;
    void ComputeRunWithFloat16() override;

 protected:
    float lr, decay, eps;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_UPDATE_RMSPROP_UPDATE_OP_H_