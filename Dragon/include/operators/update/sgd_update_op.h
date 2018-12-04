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

#ifndef DRAGON_OPERATORS_UPDATE_SGD_UPDATE_OP_H_
#define DRAGON_OPERATORS_UPDATE_SGD_UPDATE_OP_H_

#include "operators/update/update_op_base.h"

namespace dragon {

template <class Context>
class SGDUpdateOp final : public UpdateOpBase<Context> {
 public:
    SGDUpdateOp(const OperatorDef& def, Workspace* ws)
        : UpdateOpBase<Context>(def, ws),
        old_lr(-1.f), correction(1.f) {}
    USE_OPERATOR_FUNCTIONS;
    USE_UPDATER_FUNCTIONS(Context);

    void ComputeRunWithFloat32() override;
    void ComputeRunWithFloat16() override;

 protected:
    float old_lr, lr, momentum, correction;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_UPDATE_SGD_UPDATE_OP_H_