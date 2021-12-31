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

#ifndef DRAGON_OPERATORS_TRAINING_UPDATE_OP_H_
#define DRAGON_OPERATORS_TRAINING_UPDATE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class UpdateOpBase : public Operator<Context> {
 public:
  UpdateOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  virtual void GetArguments() {
    grad_scale_ = GetHyper<float>("grad_scale");
    weight_decay_ = OP_SINGLE_ARG(float, "weight_decay", -1.f);
    if (weight_decay_ < 0.f) {
      weight_decay_ = GetHyper<float>("weight_decay");
    }
    clip_norm_ = GetHyper<float>("clip_norm");
    clip_value_ = GetHyper<float>("clip_value");
  }

  void RunOnDevice() override;

  template <typename T>
  void TransformGrad(Tensor* dX);

  virtual void ApplyUpdate(Tensor* dX, Tensor* X, Tensor* Y) = 0;

  template <typename T>
  T GetHyper(const string& key);

  Tensor* GetState(const string& key);

 protected:
  int weight_index_;
  float grad_scale_, weight_decay_;
  float clip_norm_, clip_value_;
};

#define USE_UPDATE_FUNCTIONS                                       \
  using UpdateOpBase<Context>::GetHyper;                           \
  using UpdateOpBase<Context>::GetState;                           \
  void ApplyUpdate(Tensor* dX, Tensor* X, Tensor* Y) override {    \
    if (dX->template IsType<float>()) {                            \
      if (Y == nullptr) {                                          \
        DoRunWithType<float, float>(dX, X, Y);                     \
      } else if (Y->template IsType<float16>()) {                  \
        DoRunWithType<float, float16>(dX, X, Y);                   \
      } else {                                                     \
        LOG(FATAL) << MessageForUnsupported(                       \
            dtypes::to_string(Y->meta()), {"float16", "float32"}); \
      }                                                            \
    } else if (dX->template IsType<double>()) {                    \
      DoRunWithType<double, double>(dX, X, Y);                     \
    } else {                                                       \
      LOG(FATAL) << MessageForUnsupported(                         \
          dtypes::to_string(dX->meta()), {"float32", "float64"});  \
    }                                                              \
  }

template <class Context>
class MomentumSGDOp final : public UpdateOpBase<Context> {
 public:
  MomentumSGDOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = this->template GetHyper<float>("lr");
    momentum_ = this->template GetHyper<float>("momentum");
    UpdateOpBase<Context>::GetArguments();
  }

  template <typename T, typename CopyT>
  void DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y);

 protected:
  float lr_, momentum_;
};

template <class Context>
class NesterovSGDOp final : public UpdateOpBase<Context> {
 public:
  NesterovSGDOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = this->template GetHyper<float>("lr");
    momentum_ = this->template GetHyper<float>("momentum");
    UpdateOpBase<Context>::GetArguments();
  }

  template <typename T, typename CopyT>
  void DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y);

 protected:
  float lr_, momentum_;
};

template <class Context>
class RMSpropOp final : public UpdateOpBase<Context> {
 public:
  RMSpropOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = this->template GetHyper<float>("lr");
    momentum_ = this->template GetHyper<float>("momentum");
    alpha_ = this->template GetHyper<float>("alpha");
    eps_ = this->template GetHyper<float>("eps");
    UpdateOpBase<Context>::GetArguments();
  }

  template <typename T, typename CopyT>
  void DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y);

 protected:
  float lr_, momentum_, alpha_, eps_;
};

template <class Context>
class AdamOp : public UpdateOpBase<Context> {
 public:
  AdamOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws), t_(0) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = this->template GetHyper<float>("lr");
    beta1_ = this->template GetHyper<float>("beta1");
    beta2_ = this->template GetHyper<float>("beta2");
    eps_ = this->template GetHyper<float>("eps");
    t_++;
    correction_ = sqrt(1.f - pow(beta2_, t_)) / (1.f - pow(beta1_, t_));
    UpdateOpBase<Context>::GetArguments();
  }

  template <typename T, typename CopyT>
  void DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y);

 protected:
  int64_t t_;
  float lr_, beta1_, beta2_;
  float eps_, correction_;
};

template <class Context>
class AdamWOp final : public UpdateOpBase<Context> {
 public:
  AdamWOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws), t_(0) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = this->template GetHyper<float>("lr");
    beta1_ = this->template GetHyper<float>("beta1");
    beta2_ = this->template GetHyper<float>("beta2");
    eps_ = this->template GetHyper<float>("eps");
    t_++;
    correction_ = sqrt(1.f - pow(beta2_, t_)) / (1.f - pow(beta1_, t_));
    UpdateOpBase<Context>::GetArguments();
  }

  template <typename T, typename CopyT>
  void DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y);

 protected:
  int64_t t_;
  float lr_, beta1_, beta2_;
  float eps_, correction_;
};

template <class Context>
class LARSOp final : public UpdateOpBase<Context> {
 public:
  LARSOp(const OperatorDef& def, Workspace* ws)
      : UpdateOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = this->template GetHyper<float>("lr");
    momentum_ = this->template GetHyper<float>("momentum");
    trust_coef_ = this->template GetHyper<float>("trust_coef");
    UpdateOpBase<Context>::GetArguments();
  }

  template <typename T, typename CopyT>
  void DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y);

 protected:
  float lr_, momentum_, trust_coef_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_TRAINING_UPDATE_OP_H_
