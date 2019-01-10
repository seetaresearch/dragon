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

#ifndef DRAGON_OPERATORS_VISION_LRN_OP_H_
#define DRAGON_OPERATORS_VISION_LRN_OP_H_

#include "core/operator.h"

namespace dragon {

typedef enum {
    ACROSS_CHANNELS,
    WITHIN_CHANNEL,
} LRNMode;

template <class Context>
class LRNOp : public Operator<Context> {
 public:
    LRNOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          local_size(OperatorBase::Arg<int64_t>("local_size", 5)),
          alpha(OperatorBase::Arg<float>("alpha", 0.0001f)),
          beta(OperatorBase::Arg<float>("beta", 0.75f)),
          k(OperatorBase::Arg<float>("k", 2.f)),
          mode(OperatorBase::Arg<string>("mode", "ACROSS_CHANNELS")),
          data_format(OperatorBase::Arg<string>("data_format", "NCHW")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
    template <typename T> void AcrossRunWithType();
    template <typename T> void SplitRunWithType();
    template <typename T> void SquareRunWithType();
    template <typename T> void PoolRunWithType();
    template <typename T> void PowRunWithType();
    template <typename T> void ProdRunWithType();

 protected:
    int local_size;
    float alpha, beta, k;
    string mode, data_format;
    unique_ptr<OperatorBase> sqr_op, pool_op, pow_op, prod_op;
    Tensor* sqr_in, *prod_in, *sqr_out, *pool_out, *pow_out;
    Tensor* scale;
};

template <class Context>
class LRNGradientOp : public Operator<Context> {
 public:
    LRNGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          local_size(OperatorBase::Arg<int64_t>("local_size", 5)),
          alpha(OperatorBase::Arg<float>("alpha", 0.0001f)),
          beta(OperatorBase::Arg<float>("beta", 0.75f)),
          k(OperatorBase::Arg<float>("k", 2.f)),
          mode(OperatorBase::Arg<string>("mode", "ACROSS_CHANNELS")),
          data_format(OperatorBase::Arg<string>("data_format", "NCHW")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
    template <typename T> void AcrossRunWithType();
    template <typename T> void SplitRunWithType();
    template <typename T> void SquareRunWithType();
    template <typename T> void PoolRunWithType();
    template <typename T> void PowRunWithType();
    template <typename T> void ProdRunWithType();

 protected:
    int local_size;
    float alpha, beta, k;
    string mode, data_format;
    unique_ptr<OperatorBase> sqr_op, pool_op, pow_op, prod_op;
    Tensor* sqr_in, *prod_in, *sqr_out, *pool_out, *pow_out;
    Tensor* scale;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNLRNOp final : public LRNOp<Context> {
 public:
    CuDNNLRNOp(const OperatorDef& def, Workspace* ws)
        : LRNOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc));
        CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc,
            this->local_size, this->alpha, this->beta, this->k));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNLRNOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyLRNDescriptor(norm_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnLRNDescriptor_t norm_desc;
};

template <class Context>
class CuDNNLRNGradientOp final : public LRNGradientOp<Context > {
 public:
    CuDNNLRNGradientOp(const OperatorDef& def, Workspace* ws) 
        : LRNGradientOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc));
        CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc,
            this->local_size, this->alpha, this->beta, this->k));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNLRNGradientOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyLRNDescriptor(norm_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnLRNDescriptor_t norm_desc;
};

#endif  // WITH CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_LRN_OP_H_