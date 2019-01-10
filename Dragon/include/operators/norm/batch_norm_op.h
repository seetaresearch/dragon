/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_
#define DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_

#include <cfloat>

#include "core/operator.h"

namespace dragon {

template <class Context>
class BatchNormOp : public Operator<Context> {
 public:
    BatchNormOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", -1)),
          momentum(OperatorBase::Arg<float>("momentum", 0.9f)),
          eps(OperatorBase::Arg<float>("eps", 1e-5f)),
          use_stats(OperatorBase::Arg<int64_t>("use_stats", -1)) {}
    USE_OPERATOR_FUNCTIONS;

    void Reshape();

    void RunOnDevice() override;
    template <typename Tx, typename Tp> void TrainingRunWithType();
    template <typename Tx, typename Tp> void InferenceRunWithType();

 protected:
    float momentum, eps;
    string data_format;
    int64_t axis, use_stats, N, C, S;
    Tensor *mean, *var, scale, bias;
    bool is_training, is_recomputing;
};

template <class Context>
class BatchNormGradientOp : public Operator<Context> {
 public:
    BatchNormGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", -1)),
          eps(OperatorBase::Arg<float>("eps", 1e-5f)),
          use_stats(OperatorBase::Arg<int64_t>("use_stats", -1)) {}
    USE_OPERATOR_FUNCTIONS;

    void Reshape();

    void RunOnDevice() override;
    template <typename Tx, typename Tp> void TrainingRunWithType();
    template <typename Tx, typename Tp> void InferenceRunWithType();

 protected:
    float eps;
    string data_format;
    int64_t axis, use_stats, N, C, S, NC, NS;
    Tensor *mean, *var, dscale, dbias;
    bool is_training;
};

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

template <class Context>
class CuDNNBatchNormOp final : public BatchNormOp<Context> {
 public:
    CuDNNBatchNormOp(const OperatorDef& def, Workspace* ws)
        : BatchNormOp<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", -1)),
          eps64(OperatorBase::Arg<float>("eps", 1e-5f)),
          use_stats(OperatorBase::Arg<int64_t>("use_stats", -1)) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
        if (eps64 <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON)
            LOG(FATAL) << "Provided epsilon is smaller than "
                << "CUDNN_BN_MIN_EPSILON. Setting it to "
                << "CUDNN_BN_MIN_EPSILON instead.";
        eps64 = std::max(eps64, CUDNN_BN_MIN_EPSILON);
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNBatchNormOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
    }

    void Reshape();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    double eps64;
    int64_t axis, use_stats, N, C;
    string data_format;
    Tensor* mean, *var;
    bool is_training, is_recomputing;
    cudnnTensorDescriptor_t input_desc, output_desc, bn_desc;
    cudnnBatchNormMode_t bn_mode;
};

template <class Context>
class CuDNNBatchNormGradientOp final
    : public BatchNormGradientOp<Context> {
 public:
    CuDNNBatchNormGradientOp(const OperatorDef& def, Workspace* ws)
        : BatchNormGradientOp<Context>(def, ws),
        axis(OperatorBase::Arg<int64_t>("axis", -1)),
        eps64(OperatorBase::Arg<float>("eps", 1e-5f)),
        use_stats(OperatorBase::Arg<int64_t>("use_stats", -1)) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
        if (eps64 <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON)
            LOG(FATAL) << "Provided epsilon is smaller than "
            << "CUDNN_BN_MIN_EPSILON. Setting it to "
            << "CUDNN_BN_MIN_EPSILON instead.";
        eps64 = std::max(eps64, CUDNN_BN_MIN_EPSILON);
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNBatchNormGradientOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
    }

    void Reshape();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    double eps64;
    int64_t axis, use_stats, N, C, S;
    string data_format;
    Tensor* mean, *var;
    bool is_training;
    cudnnTensorDescriptor_t input_desc, output_desc, bn_desc;
    cudnnBatchNormMode_t bn_mode;
};

#endif

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_