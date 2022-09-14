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

#ifndef DRAGON_OPERATORS_NORMALIZATION_BATCH_NORM_OP_H_
#define DRAGON_OPERATORS_NORMALIZATION_BATCH_NORM_OP_H_

#include <cfloat>

#include "dragon/core/operator.h"
#include "dragon/operators/distributed/collective_op_base.h"

namespace dragon {

// Multiple inheritance is forbidden by the registry.
// So, we should inherit the collective op base if mpi available.
#ifdef USE_MPI
#define GenericOpBase CollectiveOpBase
#else
#define GenericOpBase Operator
#endif

template <class Context>
class BatchNormOpBase : public GenericOpBase<Context> {
 public:
  BatchNormOpBase(const OperatorDef& def, Workspace* ws)
      : GenericOpBase<Context>(def, ws),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-5)),
        use_stats_(OP_SINGLE_ARG(int64_t, "use_stats", -1)),
        sync_stats_(OP_SINGLE_ARG(int64_t, "comm", 0) > 0 ? 1 : 0) {}
  USE_OPERATOR_FUNCTIONS;

  void GetBaseArguments() {
    auto& X = Input(0);
    GET_OP_AXIS_ARG(axis, X.ndim(), -1);
    // Set dimensions.
    N_ = X.dim(0), C_ = X.dim(axis);
    S_ = X.count() / N_ / C_;
    // Set data format.
    this->data_format_ = "NCHW";
    if (axis + 1 == X.ndim()) this->data_format_ = "NHWC";
    // Set training mode.
    training_ = use_stats_ < 0 ? (phase() == "TRAIN" ? 1 : 0)
                               : (use_stats_ > 0 ? 0 : 1);
  }

 protected:
  double epsilon_;
  int64_t N_, C_, S_;
  int64_t use_stats_, sync_stats_;
  int64_t training_;
};

#undef GenericOpBase

#define USE_BATCHNORM_FUNCTIONS                     \
  using BatchNormOpBase<Context>::GetBaseArguments; \
  using BatchNormOpBase<Context>::epsilon_;         \
  using BatchNormOpBase<Context>::use_stats_;       \
  using BatchNormOpBase<Context>::sync_stats_;      \
  using BatchNormOpBase<Context>::N_;               \
  using BatchNormOpBase<Context>::C_;               \
  using BatchNormOpBase<Context>::S_;               \
  using BatchNormOpBase<Context>::training_

template <class Context>
class BatchNormOp : public BatchNormOpBase<Context> {
 public:
  BatchNormOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBase<Context>(def, ws) {
    INITIALIZE_OP_SINGLE_ARG(float, momentum, 0.9f);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;
#ifdef USE_MPI
  USE_COLLECTIVE_FUNCTIONS;
#endif

  void Setup() override {
    GetBaseArguments();
    Output(0)->ReshapeLike(Input(0));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void RunTraining();

  template <typename T>
  void RunInference();

  template <typename T>
  void DoRunWithType() {
    if (training_) {
      RunTraining<T>();
    } else {
      RunInference<T>();
    }
  };

  DECLARE_OP_SINGLE_ARG(float, momentum);
};

template <class Context>
class BatchNormGradientOp : public BatchNormOpBase<Context> {
 public:
  BatchNormGradientOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;
#ifdef USE_MPI
  USE_COLLECTIVE_FUNCTIONS;
#endif

  void Setup() override {
    GetBaseArguments();
    Output(0)->ReshapeLike(Input(0));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void RunTraining();

  template <typename T>
  void RunInference();

  template <typename T>
  void DoRunWithType() {
    if (training_) {
      RunTraining<T>();
    } else {
      RunInference<T>();
    }
  };
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNBatchNormOp final : public BatchNormOpBase<Context> {
 public:
  CuDNNBatchNormOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBase<Context>(def, ws) {
    CuDNNCreateTensorDesc(&bn_desc_);
    CuDNNCreateTensorDesc(&input_desc_);
    if (epsilon_ <= CUDNN_BN_MIN_EPSILON) {
      epsilon_ = CUDNN_BN_MIN_EPSILON;
    }
    INITIALIZE_OP_SINGLE_ARG(float, momentum, 0.9f);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;

  ~CuDNNBatchNormOp() {
    CuDNNDestroyTensorDesc(&bn_desc_);
    CuDNNDestroyTensorDesc(&input_desc_);
  }

  void Setup() override {
    GetBaseArguments();
    Output(0)->ReshapeLike(Input(0));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_, bn_desc_;
  cudnnBatchNormMode_t bn_mode_;
  DECLARE_OP_SINGLE_ARG(float, momentum);
};

template <class Context>
class CuDNNBatchNormGradientOp final : public BatchNormGradientOp<Context> {
 public:
  CuDNNBatchNormGradientOp(const OperatorDef& def, Workspace* ws)
      : BatchNormGradientOp<Context>(def, ws) {
    CuDNNCreateTensorDesc(&bn_desc_);
    CuDNNCreateTensorDesc(&input_desc_);
    if (epsilon_ <= CUDNN_BN_MIN_EPSILON) {
      epsilon_ = CUDNN_BN_MIN_EPSILON;
    }
  }
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;

  ~CuDNNBatchNormGradientOp() {
    CuDNNDestroyTensorDesc(&bn_desc_);
    CuDNNDestroyTensorDesc(&input_desc_);
  }

  void Setup() override {
    GetBaseArguments();
    Output(0)->ReshapeLike(Input(0));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void RunTraining();

  template <typename T>
  void DoRunWithType() {
    if (training_) {
      RunTraining<T>();
    } else {
      this->template RunInference<T>();
    }
  };

 protected:
  cudnnTensorDescriptor_t input_desc_, bn_desc_;
  cudnnBatchNormMode_t bn_mode_;
};

DEFINE_OP_SINGLE_ARG(float, CuDNNBatchNormOp, momentum);

#endif // USE_CUDNN

#ifdef USE_MPS

template <class Context>
class MPSBatchNormGradientOp : public BatchNormOpBase<Context> {
 public:
  MPSBatchNormGradientOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;
#ifdef USE_MPI
  USE_COLLECTIVE_FUNCTIONS;
#endif

  void Setup() override {
    GetBaseArguments();
    Output(0)->ReshapeLike(Input(0));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void RunTraining();

  template <typename T>
  void RunInference();

  template <typename T>
  void DoRunWithType() {
    if (training_) {
      RunTraining<T>();
    } else {
      RunInference<T>();
    }
  };
};

#endif

DEFINE_OP_SINGLE_ARG(float, BatchNormOp, momentum);

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_BATCH_NORM_OP_H_
