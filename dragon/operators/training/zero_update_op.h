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

#ifndef DRAGON_OPERATORS_TRAINING_ZERO_UPDATE_OP_H_
#define DRAGON_OPERATORS_TRAINING_ZERO_UPDATE_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/distributed/collective_op_impl.h"

namespace dragon {

template <class Context>
class ZeroUpdateOpBase : public Operator<Context> {
 public:
  ZeroUpdateOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        bucket_name_(OP_SINGLE_ARG(string, "bucket_name", "")),
        use_lr_scales_(OP_SINGLE_ARG(int64_t, "use_lr_scales", 0)) {
    coll_impl_.SetBackend(OP_SINGLE_ARG(string, "backend", "MPI"));
    coll_impl_.SetComm(
        OP_SINGLE_ARG(int64_t, "comm", 0),
        OP_SINGLE_ARG(int64_t, "group", 0),
        OP_REPEATED_ARG(int64_t, "ranks"));
    INITIALIZE_OP_REPEATED_ARG(float, lr_scales);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  T GetHyper(const string& key);

  Tensor* GetState(const string& key);

  template <typename T>
  Tensor* InitMaster();

  template <typename T>
  void SetDenseHyper(int64_t N, T* x, T* y);

  template <typename T>
  void SetLRS(int64_t N, T* y);

  template <typename T>
  void CopyBuffer(T* buf, bool cat = true);

  template <typename T>
  void DoRunWithType();

  virtual void GetArguments() {
    grad_scale_ = GetHyper<float>("grad_scale");
    weight_decay_ = OP_SINGLE_ARG(float, "weight_decay", -1.f);
    if (weight_decay_ < 0.f) {
      weight_decay_ = GetHyper<float>("weight_decay");
    }
    clip_norm_ = GetHyper<float>("clip_norm");
    clip_value_ = GetHyper<float>("clip_value");
  }

  template <typename T>
  void TransformGrad(Tensor* dX);

  virtual void ApplyUpdate(Tensor* dX, Tensor* X, Tensor* Y) = 0;

 protected:
  string bucket_name_;
  float grad_scale_, weight_decay_;
  float clip_norm_, clip_value_;
  int64_t use_lr_scales_;
  DECLARE_OP_REPEATED_ARG(float, lr_scales);
  CollectiveOpImpl coll_impl_;
};

#define USE_ZERO_UPDATE_FUNCTIONS                                             \
  using ZeroUpdateOpBase<Context>::GetState;                                  \
  void ApplyUpdate(Tensor* dX, Tensor* X, Tensor* Y) override {               \
    if (Y->template IsType<float16>()) {                                      \
      DoRunWithType<float, float16>(dX, X, Y);                                \
    } else if (Y->template IsType<bfloat16>()) {                              \
      DoRunWithType<float, bfloat16>(dX, X, Y);                               \
    } else if (Y->template IsType<float>()) {                                 \
      DoRunWithType<float, float>(dX, X, Y);                                  \
    } else {                                                                  \
      LOG(FATAL) << MessageForUnsupported(                                    \
          dtypes::to_string(dX->meta()), {"float16", "bfloat16", "float32"}); \
    }                                                                         \
  }

template <class Context>
class ZeroAdamWOp final : public ZeroUpdateOpBase<Context> {
 public:
  ZeroAdamWOp(const OperatorDef& def, Workspace* ws)
      : ZeroUpdateOpBase<Context>(def, ws), t_(0) {}
  USE_OPERATOR_FUNCTIONS;
  USE_ZERO_UPDATE_FUNCTIONS;

  void GetArguments() override {
    lr_ = this->template GetHyper<float>("lr");
    beta1_ = this->template GetHyper<float>("beta1");
    beta2_ = this->template GetHyper<float>("beta2");
    eps_ = this->template GetHyper<float>("eps");
    t_++;
    correction_ = sqrt(1.f - pow(beta2_, t_)) / (1.f - pow(beta1_, t_));
    ZeroUpdateOpBase<Context>::GetArguments();
  }

  template <typename T, typename CopyT>
  void DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y);

 protected:
  int64_t t_;
  float lr_, beta1_, beta2_;
  float eps_, correction_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_TRAINING_ZERO_UPDATE_OP_H_
