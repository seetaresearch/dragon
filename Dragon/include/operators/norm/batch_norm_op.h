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
          axis_(OpArg<int64_t>("axis", -1)),
          momentum_(OpArg<float>("momentum", 0.9f)),
          eps_(OpArg<float>("eps", 1e-5f)),
          use_stats_(OpArg<int64_t>("use_stats", -1)) {}
    USE_OPERATOR_FUNCTIONS;

    void Reshape();

    void RunOnDevice() override;
    template <typename Tx, typename Tp> void TrainingImpl();
    template <typename Tx, typename Tp> void InferenceImpl();

 protected:
    float momentum_, eps_;
    int64_t axis_, use_stats_, N_, C_, S_;
    Tensor *mean_, *var_, scale_, bias_;
    bool is_training_, is_recomp_;
};

template <class Context>
class BatchNormGradientOp : public Operator<Context> {
 public:
    BatchNormGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", -1)),
          eps_(OpArg<float>("eps", 1e-5f)),
          use_stats_(OpArg<int64_t>("use_stats", -1)) {}
    USE_OPERATOR_FUNCTIONS;

    void Reshape();

    void RunOnDevice() override;
    template <typename Tx, typename Tp> void TrainingImpl();
    template <typename Tx, typename Tp> void InferenceImpl();

 protected:
    float eps_;
    int64_t N_, C_, S_, NC_, NS_;
    int64_t axis_, use_stats_, is_training_;
    Tensor* mean_, *var_, dscale_, dbias_;
};

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

template <class Context>
class CuDNNBatchNormOp final : public BatchNormOp<Context> {
 public:
    CuDNNBatchNormOp(const OperatorDef& def, Workspace* ws)
        : BatchNormOp<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", -1)),
          eps64_(OpArg<float>("eps", 1e-5f)),
          use_stats_(OpArg<int64_t>("use_stats", -1)) {
        CuDNNCreateTensorDesc(&bn_desc_);
        CuDNNCreateTensorDesc(&input_desc_);
        if (eps64_ <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON)
            LOG(FATAL) << "Provided epsilon is smaller than "
                       << "CUDNN_BN_MIN_EPSILON. \nSet it to "
                       << "CUDNN_BN_MIN_EPSILON instead.";
        eps64_ = std::max(eps64_, CUDNN_BN_MIN_EPSILON);
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNBatchNormOp() {
        CuDNNDestroyTensorDesc(&bn_desc_);
        CuDNNDestroyTensorDesc(&input_desc_);
    }

    void Reshape();

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    double eps64_;
    int64_t axis_, N_, C_;
    int64_t use_stats_, is_training_, is_recomp_;
    Tensor* mean_, *var_;
    cudnnTensorDescriptor_t input_desc_, bn_desc_;
    cudnnBatchNormMode_t bn_mode_;
};

template <class Context>
class CuDNNBatchNormGradientOp final
    : public BatchNormGradientOp<Context> {
 public:
    CuDNNBatchNormGradientOp(const OperatorDef& def, Workspace* ws)
        : BatchNormGradientOp<Context>(def, ws),
        axis_(OpArg<int64_t>("axis", -1)),
        eps64_(OpArg<float>("eps", 1e-5f)),
        use_stats_(OpArg<int64_t>("use_stats", -1)) {
        CuDNNCreateTensorDesc(&bn_desc_);
        CuDNNCreateTensorDesc(&input_desc_);
        if (eps64_ <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON)
            LOG(FATAL) << "Provided epsilon is smaller than "
                       << "CUDNN_BN_MIN_EPSILON. \nSet it to "
                       << "CUDNN_BN_MIN_EPSILON instead.";
        eps64_ = std::max(eps64_, CUDNN_BN_MIN_EPSILON);
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNBatchNormGradientOp() {
        CuDNNDestroyTensorDesc(&bn_desc_);
        CuDNNDestroyTensorDesc(&input_desc_);
    }

    void Reshape();

    void RunOnDevice() override;
    template <typename T> void TrainingImpl();
    template <typename T> void InferenceImpl();

 protected:
    double eps64_;
    int64_t axis_, N_, C_, S_;
    int64_t use_stats_, is_training_;
    Tensor* mean_, *var_;
    cudnnTensorDescriptor_t input_desc_, bn_desc_;
    cudnnBatchNormMode_t bn_mode_;
};

#endif

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_