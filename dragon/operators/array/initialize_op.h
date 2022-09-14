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

#ifndef DRAGON_OPERATORS_ARRAY_INITIALIZE_OP_H_
#define DRAGON_OPERATORS_ARRAY_INITIALIZE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class InitializeOp : public Operator<Context> {
 public:
  InitializeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, dims);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, dims);
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
class RangeOp final : public InitializeOp<Context> {
 public:
  RangeOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(double, slice);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(double, slice);
};

template <class Context>
class LinSpaceOp final : public InitializeOp<Context> {
 public:
  LinSpaceOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(double, start);
    INITIALIZE_OP_REPEATED_ARG(double, stop);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(double, start);
  DECLARE_OP_REPEATED_ARG(double, stop);
};

template <class Context>
class PermutationOp final : public InitializeOp<Context> {
 public:
  PermutationOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws) {
    INITIALIZE_OP_SINGLE_ARG(int64_t, limit, 0);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG(int64_t, limit);
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
      : InitializeOp<Context>(def, ws),
        mean_(OP_SINGLE_ARG(float, "mean", 0.f)),
        std_(OP_SINGLE_ARG(float, "std", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float mean_, std_;
};

template <class Context>
class RandomUniformOp final : public InitializeOp<Context> {
 public:
  RandomUniformOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws),
        low_(OP_SINGLE_ARG(float, "low", 0.f)),
        high_(OP_SINGLE_ARG(float, "high", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float low_, high_;
};

template <class Context>
class TruncatedNormalOp final : public InitializeOp<Context> {
 public:
  TruncatedNormalOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws),
        mean_(OP_SINGLE_ARG(float, "mean", 0.f)),
        std_(OP_SINGLE_ARG(float, "std", 1.f)) {
    low_ = OP_SINGLE_ARG(float, "low", mean_ - 2 * std_);
    high_ = OP_SINGLE_ARG(float, "high", mean_ + 2 * std_);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float mean_, std_, low_, high_;
};

template <class Context>
class GlorotNormalOp final : public InitializeOp<Context> {
 public:
  GlorotNormalOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws),
        scale_(OP_SINGLE_ARG(float, "scale", 2.f)),
        mode_(OP_SINGLE_ARG(string, "mode", "fan_in")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float scale_;
  string mode_;
};

template <class Context>
class GlorotUniformOp final : public InitializeOp<Context> {
 public:
  GlorotUniformOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws),
        scale_(OP_SINGLE_ARG(float, "scale", 3.f)),
        mode_(OP_SINGLE_ARG(string, "mode", "fan_in")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float scale_;
  string mode_;
};

DEFINE_OP_SINGLE_ARG(int64_t, PermutationOp, limit);
DEFINE_OP_REPEATED_ARG(int64_t, InitializeOp, dims);
DEFINE_OP_REPEATED_ARG(double, RangeOp, slice);
DEFINE_OP_REPEATED_ARG(double, LinSpaceOp, start);
DEFINE_OP_REPEATED_ARG(double, LinSpaceOp, stop);

#ifdef USE_MPS

template <class Context>
class MPSRandomUniformOp final : public InitializeOp<Context> {
 public:
  MPSRandomUniformOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws),
        low_(OP_SINGLE_ARG(float, "low", 0.f)),
        high_(OP_SINGLE_ARG(float, "high", 1.f)) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSRandomUniformOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float low_, high_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSRandomNormalOp final : public InitializeOp<Context> {
 public:
  MPSRandomNormalOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws),
        mean_(OP_SINGLE_ARG(float, "mean", 0.f)),
        std_(OP_SINGLE_ARG(float, "std", 1.f)) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSRandomNormalOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float mean_, std_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSTruncatedNormalOp final : public InitializeOp<Context> {
 public:
  MPSTruncatedNormalOp(const OperatorDef& def, Workspace* ws)
      : InitializeOp<Context>(def, ws),
        mean_(OP_SINGLE_ARG(float, "mean", 0.f)),
        std_(OP_SINGLE_ARG(float, "std", 1.f)) {
    low_ = OP_SINGLE_ARG(float, "low", mean_ - 2 * std_);
    high_ = OP_SINGLE_ARG(float, "high", mean_ + 2 * std_);
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSTruncatedNormalOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float mean_, std_, low_, high_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_INITIALIZE_OP_H_
