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
    N_ = X.dim(0), C_ = X.dim(axis);
    S_ = X.count() / N_ / C_;
    this->data_format_ = "NCHW";
    if (axis + 1 == X.ndim()) this->data_format_ = "NHWC";
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
    if (epsilon_ <= CUDNN_BN_MIN_EPSILON) epsilon_ = CUDNN_BN_MIN_EPSILON;
    INITIALIZE_OP_SINGLE_ARG(float, momentum, 0.9f);
    CuDNNCreateTensorDesc(&bn_desc_);
    CuDNNCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;

  ~CuDNNBatchNormOp() {
    CuDNNDestroyTensorDesc(bn_desc_);
    CuDNNDestroyTensorDesc(input_desc_);
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
  cudnnBatchNormMode_t bn_mode_;
  cudnnTensorDescriptor_t input_desc_, bn_desc_;
  DECLARE_OP_SINGLE_ARG(float, momentum);
};

template <class Context>
class CuDNNBatchNormGradientOp final : public BatchNormGradientOp<Context> {
 public:
  CuDNNBatchNormGradientOp(const OperatorDef& def, Workspace* ws)
      : BatchNormGradientOp<Context>(def, ws) {
    if (epsilon_ <= CUDNN_BN_MIN_EPSILON) epsilon_ = CUDNN_BN_MIN_EPSILON;
    CuDNNCreateTensorDesc(&bn_desc_);
    CuDNNCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;

  ~CuDNNBatchNormGradientOp() {
    CuDNNDestroyTensorDesc(bn_desc_);
    CuDNNDestroyTensorDesc(input_desc_);
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
  cudnnBatchNormMode_t bn_mode_;
  cudnnTensorDescriptor_t input_desc_, bn_desc_;
};
#endif // USE_CUDNN

#ifdef USE_MLU
template <class Context>
class CNNLBatchNormOp final : public BatchNormOpBase<Context> {
 public:
  CNNLBatchNormOp(const OperatorDef& def, Workspace* ws)
      : BatchNormOpBase<Context>(def, ws) {
    INITIALIZE_OP_SINGLE_ARG(float, momentum, 0.9f);
    CNNLCreateTensorDesc(&bn_desc_);
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&syncbn_desc_);
    CNNLCreateTensorDesc(&count_desc_);
    CNNL_CHECK(cnnlCreateActivationDescriptor(&act_desc_));
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(
        act_desc_,
        CNNL_ACTIVATION_IDENTITY,
        CNNL_ACTIVATION_FAST,
        CNNL_PROPAGATE_NAN,
        0.f,
        0,
        1.f, // gamma
        1.f, // scale
        true,
        false));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;
#ifdef USE_MPI
  USE_COLLECTIVE_FUNCTIONS;
#endif

  ~CNNLBatchNormOp() {
    CNNLDestroyTensorDesc(bn_desc_);
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(syncbn_desc_);
    CNNLDestroyTensorDesc(count_desc_);
    CNNL_CHECK(cnnlDestroyActivationDescriptor(act_desc_));
  }

  void Setup() override {
    GetBaseArguments();
    CHECK_EQ(data_format(), "NHWC");
    Output(0)->ReshapeLike(Input(0));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlBatchNormMode_t bn_mode_;
  cnnlActivationDescriptor_t act_desc_;
  cnnlTensorDescriptor_t input_desc_, bn_desc_;
  cnnlTensorDescriptor_t syncbn_desc_, count_desc_;
  DECLARE_OP_SINGLE_ARG(float, momentum);
};

template <class Context>
class CNNLBatchNormGradientOp final : public BatchNormGradientOp<Context> {
 public:
  CNNLBatchNormGradientOp(const OperatorDef& def, Workspace* ws)
      : BatchNormGradientOp<Context>(def, ws) {
    CNNLCreateTensorDesc(&bn_desc_);
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&syncbn_desc_);
    CNNLCreateTensorDesc(&count_desc_);
    CNNL_CHECK(cnnlCreateActivationDescriptor(&act_desc_));
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(
        act_desc_,
        CNNL_ACTIVATION_IDENTITY,
        CNNL_ACTIVATION_FAST,
        CNNL_PROPAGATE_NAN,
        0.f,
        0,
        1.f, // gamma
        1.f, // scale
        true,
        false));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_BATCHNORM_FUNCTIONS;
#ifdef USE_MPI
  USE_COLLECTIVE_FUNCTIONS;
#endif

  ~CNNLBatchNormGradientOp() {
    CNNLDestroyTensorDesc(bn_desc_);
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(syncbn_desc_);
    CNNLDestroyTensorDesc(count_desc_);
    CNNL_CHECK(cnnlDestroyActivationDescriptor(act_desc_));
  }

  void Setup() override {
    GetBaseArguments();
    CHECK_EQ(data_format(), "NHWC");
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
      NOT_IMPLEMENTED;
    }
  };

 protected:
  cnnlBatchNormMode_t bn_mode_;
  cnnlActivationDescriptor_t act_desc_;
  cnnlTensorDescriptor_t input_desc_, bn_desc_;
  cnnlTensorDescriptor_t syncbn_desc_, count_desc_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_BATCH_NORM_OP_H_
