/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_TRAINING_SGD_UPDATE_OP_H_
#define DRAGON_OPERATORS_TRAINING_SGD_UPDATE_OP_H_

#include "dragon/operators/training/update_op_base.h"

namespace dragon {

template <class Context>
class SGDUpdateOp final : public UpdateOpBase<Context> {
 public:
  SGDUpdateOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws), last_lr_(-1.f), correction_(1.f) {}
  USE_OPERATOR_FUNCTIONS;
  USE_PARAM_UPDATE_FUNCTIONS;

  void Compute(Tensor* dX) override;

 protected:
  float last_lr_, correction_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_TRAINING_SGD_UPDATE_OP_H_
