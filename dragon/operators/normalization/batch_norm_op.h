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

  void DetermineBaseArguments() {
    auto& X = Input(0);
    // Determine the training mode
    if (use_stats_ == -1) {
      is_training_ = phase() == "TRAIN" ? 1 : 0;
    } else {
      is_training_ = use_stats_ > 0 ? 0 : 1;
    }
    // Determine the data format
    this->data_format_ = "NCHW";
    auto axis = OP_SINGLE_ARG(int64_t, "axis", -1);
    if (axis == -1) axis += X.ndim();
    if (axis + 1 == X.ndim()) this->data_format_ = "NHWC";
    N_ = X.dim(0), C_ = X.dim(axis);
    S_ = X.count() / N_ / C_;
  }

 protected:
  double epsilon_;
  int64_t N_, C_, S_;
  int64_t use_stats_, sync_stats_;
  int64_t is_training_, is_recomputing_;
};

#undef GenericOpBase

#define USE_BATCHNORM_FUNCTIONS                           \
  using BatchNormOpBase<Context>::DetermineBaseArguments; \
  using BatchNormOpBase<Context>::epsilon_;               \
  using BatchNormOpBase<Context>::use_stats_;             \
  using BatchNormOpBase<Context>::sync_stats_;            \
  using BatchNormOpBase<Context>::N_;                     \
  using BatchNormOpBase<Context>::C_;                     \
  using BatchNormOpBase<Context>::S_;                     \
  using BatchNormOpBase<Context>::is_training_;           \
  using BatchNormOpBase<Context>::is_recomputing_

template <class Context>
class BatchNormOp : public BatchNormOpBase<Context> {
 public:
  BatchNormOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBase<Context>(def, ws) {
    INIT_OP_SINGLE_ARG_WITH_DESC(float, momentum, 0.9f);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;
#ifdef USE_MPI
  USE_COLLECTIVE_FUNCTIONS;
#endif

  void RunOnDevice() override;

  template <typename T>
  void TrainingImpl();

  template <typename T>
  void InferenceImpl();

  template <typename T>
  void DoRunWithType() {
    if (is_training_) {
      TrainingImpl<T>();
    } else {
      InferenceImpl<T>();
    }
  };

  DECLARE_OP_SINGLE_ARG_WITH_DESC(float, momentum);
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

  void RunOnDevice() override;

  template <typename T>
  void TrainingImpl();

  template <typename T>
  void InferenceImpl();

  template <typename T>
  void DoRunWithType() {
    if (is_training_) {
      TrainingImpl<T>();
    } else {
      InferenceImpl<T>();
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
    INIT_OP_SINGLE_ARG_WITH_DESC(float, momentum, 0.9f);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;

  ~CuDNNBatchNormOp() {
    CuDNNDestroyTensorDesc(&bn_desc_);
    CuDNNDestroyTensorDesc(&input_desc_);
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_, bn_desc_;
  cudnnBatchNormMode_t bn_mode_;
  DECLARE_OP_SINGLE_ARG_WITH_DESC(float, momentum);
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

  void RunOnDevice() override;

  template <typename T>
  void TrainingImpl();

  template <typename T>
  void DoRunWithType() {
    if (is_training_) {
      TrainingImpl<T>();
    } else {
      this->template InferenceImpl<T>();
    }
  };

 protected:
  cudnnTensorDescriptor_t input_desc_, bn_desc_;
  cudnnBatchNormMode_t bn_mode_;
};

DEFINE_OP_SINGLE_ARG_WITH_DESC(float, CuDNNBatchNormOp, momentum);

#endif // USE_CUDNN

DEFINE_OP_SINGLE_ARG_WITH_DESC(float, BatchNormOp, momentum);

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_BATCH_NORM_OP_H_
