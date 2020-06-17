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

#ifndef DRAGON_OPERATORS_ARRAY_INITIALIZE_OPS_H_
#define DRAGON_OPERATORS_ARRAY_INITIALIZE_OPS_H_

#include "dragon/core/operator.h"
#include "dragon/utils/filler.h"

namespace dragon {

template <class Context>
class InitializeOp : public Operator<Context> {
 public:
  InitializeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    GET_ARGS_WITH_DESC(int64_t, dims);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

 protected:
  TensorFillerProto proto_;
  DECLARE_ARGS_WITH_DESC(int64_t, dims);
};

template <class Context>
class FillOp final : public InitializeOp<Context> {
 public:
  FillOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws), value_(OpArg<float>("value", 0.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float value_;
};

template <class Context>
class ArangeOp final : public Operator<Context> {
 public:
  ArangeOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {
    GET_ARGS_WITH_DESC(float, slice);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_ARGS_WITH_DESC(float, slice);
};

template <class Context>
class EyeOp final : public InitializeOp<Context> {
 public:
  EyeOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws), k_(OpArg<int64_t>("k", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t k_;
};

namespace {

template <typename T>
struct TypeIdentity {
  typedef T type;
};

} // namespace

template <class Context>
class GivenTensorFillOp final : public Operator<Context> {
 public:
  GivenTensorFillOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), shape_(OpArgs<int64_t>("shape")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void Extract() {
    ExtractImpl(TypeIdentity<T>());
  }

  template <typename T>
  void ExtractImpl(TypeIdentity<T>) {
    auto raw_values = OpArgs<T>("values");
    auto nelements = (int64_t)raw_values.size();
    values_.Reshape({nelements});
    memcpy(
        values_.template mutable_data<T, CPUContext>(),
        raw_values.data(),
        nelements * sizeof(T));
  }

  void ExtractImpl(TypeIdentity<float16>) {
    auto raw_values = OpArgs<float>("values");
    auto nelements = (int64_t)raw_values.size();
    values_.Reshape({nelements});
    memcpy(
        values_.template mutable_data<float16, CPUContext>(),
        raw_values.data(),
        nelements * sizeof(float16));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  Tensor values_;
  vector<int64_t> shape_;
};

template <class Context>
class RandomNormalOp final : public InitializeOp<Context> {
 public:
  RandomNormalOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws) {
    auto mu = OpArg<float>("mean", 0.f);
    auto sigma = OpArg<float>("std", 1.f);
    this->proto_.set_mean(mu);
    this->proto_.set_std(sigma);
    this->proto_.set_type("normal");
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class RandomUniformOp final : public InitializeOp<Context> {
 public:
  RandomUniformOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws) {
    auto low = OpArg<float>("low", -1.f);
    auto high = OpArg<float>("high", 1.f);
    this->proto_.set_low(low);
    this->proto_.set_high(high);
    this->proto_.set_type("uniform");
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class TruncatedNormalOp final : public InitializeOp<Context> {
 public:
  TruncatedNormalOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws) {
    auto mu = OpArg<float>("mean", 0.f);
    auto sigma = OpArg<float>("std", 1.f);
    this->proto_.set_mean(mu);
    this->proto_.set_std(sigma);
    this->proto_.set_low(mu - 2 * sigma);
    this->proto_.set_high(mu + 2 * sigma);
    this->proto_.set_type("truncated_normal");
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class GlorotNormalOp final : public InitializeOp<Context> {
 public:
  GlorotNormalOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws) {
    auto scale = OpArg<float>("scale", 2.f);
    auto mode = OpArg<string>("mode", "fan_in");
    this->proto_.set_type("msra");
    if (mode == "fan_avg") {
      this->proto_.set_variance_norm(TensorFillerProto_VarianceNorm_FAN_AVG);
    } else if (mode == "fan_out") {
      this->proto_.set_variance_norm(TensorFillerProto_VarianceNorm_FAN_OUT);
    } else {
      this->proto_.set_variance_norm(TensorFillerProto_VarianceNorm_FAN_IN);
    }
    this->proto_.set_scale(scale);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class GlorotUniformOp final : public InitializeOp<Context> {
 public:
  GlorotUniformOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws) {
    auto scale = OpArg<float>("scale", 3.f);
    auto mode = OpArg<string>("mode", "fan_in");
    this->proto_.set_type("xavier");
    if (mode == "fan_avg") {
      this->proto_.set_variance_norm(TensorFillerProto_VarianceNorm_FAN_AVG);
    } else if (mode == "fan_out") {
      this->proto_.set_variance_norm(TensorFillerProto_VarianceNorm_FAN_OUT);
    } else {
      this->proto_.set_variance_norm(TensorFillerProto_VarianceNorm_FAN_IN);
    }
    this->proto_.set_scale(scale);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

DEFINE_ARGS_WITH_DESC(int64_t, InitializeOp, dims);
DEFINE_ARGS_WITH_DESC(float, ArangeOp, slice);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_INITIALIZE_OPS_H_
