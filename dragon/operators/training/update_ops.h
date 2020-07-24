/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_TRAINING_UPDATE_OPS_H_
#define DRAGON_OPERATORS_TRAINING_UPDATE_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class UpdateOpBase : public Operator<Context> {
 public:
  UpdateOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        lr_mult_(OpArg<float>("lr_mult", 1.f)),
        decay_mult_(OpArg<float>("decay_mult", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  virtual void ComputeUpdate(Tensor* dX) = 0;

  template <typename T>
  void AdjustGradient(Tensor* dX, Tensor* X);

  template <typename T>
  void ApplyUpdate(Tensor* dX, Tensor* X);

  Tensor* Slot(const string& name);
  float Parameter(const string& name) const;

 protected:
  float lr_mult_, decay_mult_;
};

#define USE_PARAM_UPDATE_FUNCTIONS   \
  using UpdateOpBase<Context>::Slot; \
  using UpdateOpBase<Context>::Parameter

template <class Context>
class SGDUpdateOp final : public UpdateOpBase<Context> {
 public:
  SGDUpdateOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws), last_lr_(-1.f), correction_(1.f) {}
  USE_OPERATOR_FUNCTIONS;
  USE_PARAM_UPDATE_FUNCTIONS;

  void ComputeUpdate(Tensor* dX) override;

 protected:
  float last_lr_, correction_;
};

template <class Context>
class NesterovUpdateOp final : public UpdateOpBase<Context> {
 public:
  NesterovUpdateOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_PARAM_UPDATE_FUNCTIONS;

  void ComputeUpdate(Tensor* dX) override;
};

template <class Context>
class RMSpropUpdateOp final : public UpdateOpBase<Context> {
 public:
  RMSpropUpdateOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_PARAM_UPDATE_FUNCTIONS;

  void ComputeUpdate(Tensor* dX) override;
};

template <class Context>
class AdamUpdateOp final : public UpdateOpBase<Context> {
 public:
  AdamUpdateOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws), t_(0) {}
  USE_OPERATOR_FUNCTIONS;
  USE_PARAM_UPDATE_FUNCTIONS;

  void ComputeUpdate(Tensor* dX) override;

 protected:
  int t_;
};

#undef USE_PARAM_UPDATE_FUNCTIONS

} // namespace dragon

#endif // DRAGON_OPERATORS_TRAINING_UPDATE_OPS_H_
