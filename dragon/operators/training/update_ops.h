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
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  virtual void GetArguments() {
    scale_ = Hyper("scale");
    clip_norm_ = Hyper("clip_norm");
    weight_decay_ = OP_SINGLE_ARG(float, "weight_decay", -1.f);
    if (weight_decay_ < 0.f) {
      weight_decay_ = Hyper("weight_decay");
    }
  }

  void RunOnDevice() override;

  virtual void ComputeUpdate(Tensor* dX) = 0;

  template <typename T>
  void AdjustGradient(Tensor* dX, Tensor* X);

  template <typename T>
  void ApplyUpdate(Tensor* dX, Tensor* X);

  float Hyper(const string& name);

  Tensor* Slot(const string& name);

 protected:
  int64_t input_index_;
  float scale_, clip_norm_, weight_decay_;
};

#define USE_UPDATE_FUNCTIONS          \
  using UpdateOpBase<Context>::Hyper; \
  using UpdateOpBase<Context>::Slot

template <class Context>
class SGDUpdateOp final : public UpdateOpBase<Context> {
 public:
  SGDUpdateOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws), last_lr_(-1.f), correction_(1.f) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = Hyper("lr");
    momentum_ = Hyper("momentum");
    // Momentum Correction, See arXiv:1706.02677
    if (last_lr_ > 0.f) {
      correction_ = lr_ / last_lr_;
    }
    last_lr_ = lr_; // Record the last value
    UpdateOpBase<Context>::GetArguments();
  }

  void ComputeUpdate(Tensor* dX) override;

 protected:
  float lr_, last_lr_;
  float momentum_, correction_;
};

template <class Context>
class NesterovUpdateOp final : public UpdateOpBase<Context> {
 public:
  NesterovUpdateOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = Hyper("lr");
    momentum_ = Hyper("momentum");
    UpdateOpBase<Context>::GetArguments();
  }

  void ComputeUpdate(Tensor* dX) override;

 protected:
  float lr_, momentum_;
};

template <class Context>
class RMSpropUpdateOp final : public UpdateOpBase<Context> {
 public:
  RMSpropUpdateOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = Hyper("lr");
    momentum_ = Hyper("momentum");
    decay_ = Hyper("decay");
    eps_ = Hyper("eps");
    UpdateOpBase<Context>::GetArguments();
  }

  void ComputeUpdate(Tensor* dX) override;

 protected:
  float lr_, momentum_, decay_, eps_;
};

template <class Context>
class AdamUpdateOp final : public UpdateOpBase<Context> {
 public:
  AdamUpdateOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws), t_(0) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    t_++;
    beta1_ = Hyper("beta1");
    beta2_ = Hyper("beta2");
    auto correction = sqrt(1.f - pow(beta2_, t_)) / (1.f - pow(beta1_, t_));
    lr_ = Hyper("lr") * correction;
    eps_ = Hyper("eps");
    UpdateOpBase<Context>::GetArguments();
  }

  void ComputeUpdate(Tensor* dX) override;

 protected:
  float lr_, beta1_, beta2_, eps_, t_;
};

#undef USE_UPDATE_FUNCTIONS

} // namespace dragon

#endif // DRAGON_OPERATORS_TRAINING_UPDATE_OPS_H_
