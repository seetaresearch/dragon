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
    INIT_OP_REPEATED_ARG_WITH_DESC(int64_t, dims);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

 protected:
  FillerInfo filler_info_;
  DECLARE_OP_REPEATED_ARG_WITH_DESC(int64_t, dims);
};

template <class Context>
class FillOp final : public InitializeOp<Context> {
 public:
  FillOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws),
        value_(OP_SINGLE_ARG(float, "value", 0.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float value_;
};

template <class Context>
class RangeOp final : public Operator<Context> {
 public:
  RangeOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {
    INIT_OP_REPEATED_ARG_WITH_DESC(double, slice);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG_WITH_DESC(double, slice);
};

template <class Context>
class LinSpaceOp final : public InitializeOp<Context> {
 public:
  LinSpaceOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws) {
    INIT_OP_REPEATED_ARG_WITH_DESC(double, start);
    INIT_OP_REPEATED_ARG_WITH_DESC(double, stop);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG_WITH_DESC(double, start);
  DECLARE_OP_REPEATED_ARG_WITH_DESC(double, stop);
};

template <class Context>
class PermutationOp final : public Operator<Context> {
 public:
  PermutationOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INIT_OP_SINGLE_ARG_WITH_DESC(int64_t, limit, 0);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG_WITH_DESC(int64_t, limit);
};

template <class Context>
class EyeOp final : public InitializeOp<Context> {
 public:
  EyeOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws), k_(OP_SINGLE_ARG(int64_t, "k", 0)) {}
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
      : Operator<Context>(def, ws), shape_(OP_REPEATED_ARG(int64_t, "shape")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void Extract() {
    ExtractImpl(TypeIdentity<T>());
  }

  template <typename T>
  void ExtractImpl(TypeIdentity<T>) {
    auto raw_values = OP_REPEATED_ARG(T, "values");
    auto nelements = (int64_t)raw_values.size();
    values_.Reshape({nelements});
    memcpy(
        values_.template mutable_data<T, CPUContext>(),
        raw_values.data(),
        nelements * sizeof(T));
  }

  void ExtractImpl(TypeIdentity<float16>) {
    auto raw_values = OP_REPEATED_ARG(float, "values");
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
    auto mu = OP_SINGLE_ARG(float, "mean", 0.f);
    auto sigma = OP_SINGLE_ARG(float, "std", 1.f);
    this->filler_info_.set_mean(mu);
    this->filler_info_.set_std(sigma);
    this->filler_info_.set_type("normal");
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
    auto low = OP_SINGLE_ARG(float, "low", -1.f);
    auto high = OP_SINGLE_ARG(float, "high", 1.f);
    this->filler_info_.set_low(low);
    this->filler_info_.set_high(high);
    this->filler_info_.set_type("uniform");
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
    auto mu = OP_SINGLE_ARG(float, "mean", 0.f);
    auto sigma = OP_SINGLE_ARG(float, "std", 1.f);
    this->filler_info_.set_mean(mu);
    this->filler_info_.set_std(sigma);
    this->filler_info_.set_low(mu - 2 * sigma);
    this->filler_info_.set_high(mu + 2 * sigma);
    this->filler_info_.set_type("truncated_normal");
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
    auto scale = OP_SINGLE_ARG(float, "scale", 2.f);
    auto mode = OP_SINGLE_ARG(string, "mode", "fan_in");
    this->filler_info_.set_type("glorot_normal");
    if (mode == "fan_avg") {
      this->filler_info_.set_variance_norm(FillerInfo_VarianceNorm_FAN_AVG);
    } else if (mode == "fan_out") {
      this->filler_info_.set_variance_norm(FillerInfo_VarianceNorm_FAN_OUT);
    } else {
      this->filler_info_.set_variance_norm(FillerInfo_VarianceNorm_FAN_IN);
    }
    this->filler_info_.set_scale(scale);
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
    auto scale = OP_SINGLE_ARG(float, "scale", 3.f);
    auto mode = OP_SINGLE_ARG(string, "mode", "fan_in");
    this->filler_info_.set_type("glorot_uniform");
    if (mode == "fan_avg") {
      this->filler_info_.set_variance_norm(FillerInfo_VarianceNorm_FAN_AVG);
    } else if (mode == "fan_out") {
      this->filler_info_.set_variance_norm(FillerInfo_VarianceNorm_FAN_OUT);
    } else {
      this->filler_info_.set_variance_norm(FillerInfo_VarianceNorm_FAN_IN);
    }
    this->filler_info_.set_scale(scale);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

DEFINE_OP_SINGLE_ARG_WITH_DESC(int64_t, PermutationOp, limit);
DEFINE_OP_REPEATED_ARG_WITH_DESC(int64_t, InitializeOp, dims);
DEFINE_OP_REPEATED_ARG_WITH_DESC(double, RangeOp, slice);
DEFINE_OP_REPEATED_ARG_WITH_DESC(double, LinSpaceOp, start);
DEFINE_OP_REPEATED_ARG_WITH_DESC(double, LinSpaceOp, stop);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_INITIALIZE_OPS_H_
