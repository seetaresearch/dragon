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

#ifndef DRAGON_OPERATORS_NORMALIZATION_BATCH_NORM_OP_H_
#define DRAGON_OPERATORS_NORMALIZATION_BATCH_NORM_OP_H_

#include <cfloat>

#include "dragon/core/operator.h"
#include "dragon/operators/distributed/collective_op_base.h"

namespace dragon {

// Multiple inheritance is forbidden by the registry.
// So, we should inherit the collective base as the meta.
#ifdef USE_MPI
#define BatchNormOpBaseMeta CollectiveOpBase
#else
#define BatchNormOpBaseMeta Operator
#endif

template <class Context>
class BatchNormOpBase : public BatchNormOpBaseMeta<Context> {
 public:
  BatchNormOpBase(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBaseMeta<Context>(def, ws),
        momentum_(OpArg<float>("momentum", 0.9f)),
        eps_(OpArg<float>("eps", 1e-5f)),
        use_stats_(OpArg<int64_t>("use_stats", -1)) {}
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
    auto axis = OpArg<int64_t>("axis", -1);
    if (axis == -1) axis += X.ndim();
    if (axis + 1 == X.ndim()) this->data_format_ = "NHWC";
    N_ = X.dim(0), C_ = X.dim(axis);
    S_ = X.count() / N_ / C_;
  }

 protected:
  float momentum_, eps_;
  int64_t use_stats_, N_, C_, S_;
  int64_t is_training_, is_recomputing_;
};

#undef BatchNormOpBaseMeta

#define USE_BATCHNORM_FUNCTIONS                           \
  using BatchNormOpBase<Context>::DetermineBaseArguments; \
  using BatchNormOpBase<Context>::momentum_;              \
  using BatchNormOpBase<Context>::eps_;                   \
  using BatchNormOpBase<Context>::use_stats_;             \
  using BatchNormOpBase<Context>::N_;                     \
  using BatchNormOpBase<Context>::C_;                     \
  using BatchNormOpBase<Context>::S_;                     \
  using BatchNormOpBase<Context>::is_training_;           \
  using BatchNormOpBase<Context>::is_recomputing_

template <class Context>
class BatchNormOp : public BatchNormOpBase<Context> {
 public:
  BatchNormOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;

  void RunOnDevice() override;

  template <typename InputType, typename ParamType>
  void TrainingImpl();

  template <typename InputType, typename ParamType>
  void InferenceImpl();
};

template <class Context>
class BatchNormGradientOp : public BatchNormOpBase<Context> {
 public:
  BatchNormGradientOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;

  void RunOnDevice() override;

  template <typename InputType, typename ParamType>
  void TrainingImpl();

  template <typename InputType, typename ParamType>
  void InferenceImpl();
};

#ifdef USE_MPI

template <class Context>
class SyncBatchNormOp : public BatchNormOp<Context> {
 public:
  SyncBatchNormOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;
  USE_COLLECTIVE_FUNCTIONS;

  void RunOnDevice() override;

  template <typename InputType, typename ParamType>
  void TrainingImpl();
};

template <class Context>
class SyncBatchNormGradientOp : public BatchNormGradientOp<Context> {
 public:
  SyncBatchNormGradientOp(const OperatorDef& def, Workspace* ws)
      : BatchNormGradientOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;
  USE_COLLECTIVE_FUNCTIONS;

  void RunOnDevice() override;

  template <typename InputType, typename ParamType>
  void TrainingImpl();
};

#endif // USE_MPI

#ifdef USE_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

template <class Context>
class CuDNNBatchNormOp final : public BatchNormOpBase<Context> {
 public:
  CuDNNBatchNormOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBase<Context>(def, ws), eps64_(OpArg<float>("eps", 1e-5f)) {
    CuDNNCreateTensorDesc(&bn_desc_);
    CuDNNCreateTensorDesc(&input_desc_);
    if (eps64_ <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON)
      LOG(FATAL) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. \nSet it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    eps64_ = std::max(eps64_, CUDNN_BN_MIN_EPSILON);
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
  double eps64_;
  cudnnTensorDescriptor_t input_desc_, bn_desc_;
  cudnnBatchNormMode_t bn_mode_;
};

template <class Context>
class CuDNNBatchNormGradientOp final : public BatchNormGradientOp<Context> {
 public:
  CuDNNBatchNormGradientOp(const OperatorDef& def, Workspace* ws)
      : BatchNormGradientOp<Context>(def, ws),
        eps64_(OpArg<float>("eps", 1e-5f)) {
    CuDNNCreateTensorDesc(&bn_desc_);
    CuDNNCreateTensorDesc(&input_desc_);
    if (eps64_ <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON)
      LOG(FATAL) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. \nSet it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    eps64_ = std::max(eps64_, CUDNN_BN_MIN_EPSILON);
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

 protected:
  double eps64_;
  cudnnTensorDescriptor_t input_desc_, bn_desc_;
  cudnnBatchNormMode_t bn_mode_;
};

#endif // CUDNN_VERSION_MIN(5, 0, 0)

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_BATCH_NORM_OP_H_
