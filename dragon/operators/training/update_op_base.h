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

#ifndef DRAGON_OPERATORS_TRAINING_UPDATE_OP_BASE_H_
#define DRAGON_OPERATORS_TRAINING_UPDATE_OP_BASE_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class UpdateOpBase : public Operator<Context> {
 public:
  UpdateOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        lr_mult_(OpArg<float>("lr_mult", 1.f)),
        decay_mult_(OpArg<float>("decay_mult", 1.f)),
        slot_(OpArg<string>("slot", "")) {
    CHECK(!slot_.empty()) << "\nRequired a non-empty slot";
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  virtual void Compute(Tensor* dX) = 0;

  template <typename T>
  void Process(Tensor* dX, Tensor* X);

  template <typename T>
  void Apply(Tensor* dX, Tensor* X);

  string slot() {
    return slot_ + "/" + Output(0)->name();
  }

  float param(const string& name) const;

  float lr_mult() const {
    return lr_mult_;
  }

 protected:
  string slot_;
  float lr_mult_, decay_mult_;
};

#define USE_PARAM_UPDATE_FUNCTIONS    \
  using UpdateOpBase<Context>::slot;  \
  using UpdateOpBase<Context>::param; \
  using UpdateOpBase<Context>::lr_mult

} // namespace dragon

#endif // DRAGON_OPERATORS_TRAINING_UPDATE_OP_BASE_H_
