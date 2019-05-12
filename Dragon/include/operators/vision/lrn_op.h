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
          local_size_(OpArg<int64_t>(
              "local_size", 5)),
          alpha_(OpArg<float>(
              "alpha", 0.0001f)),
          beta_(OpArg<float>(
              "beta", 0.75f)),
          k_(OpArg<float>(
              "k", 2.f)),
          mode_(OpArg<string>(
              "mode", "ACROSS_CHANNELS")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void AcrossRunImpl();
    template <typename T> void SplitRunImpl();
    template <typename T> void SquareRunImpl();
    template <typename T> void PoolRunImpl();
    template <typename T> void PowRunImpl();
    template <typename T> void ProdRunImpl();

 protected:
    string mode_;
    int local_size_;
    float alpha_, beta_, k_;
    unique_ptr<OperatorBase> sqr_op_, pool_op_;
    unique_ptr<OperatorBase> pow_op_, prod_op_;
    Tensor *scale, *sqr_in_, *prod_in_;
    Tensor *sqr_out_, *pool_out_, *pow_out_;
};

template <class Context>
class LRNGradientOp : public Operator<Context> {
 public:
    LRNGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          local_size_(OpArg<int64_t>(
              "local_size", 5)),
          alpha_(OpArg<float>(
              "alpha", 0.0001f)),
          beta_(OpArg<float>(
              "beta", 0.75f)),
          k_(OpArg<float>(
              "k", 2.f)),
          mode_(OpArg<string>(
              "mode", "ACROSS_CHANNELS")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void AcrossRunImpl();
    template <typename T> void SplitRunImpl();
    template <typename T> void SquareRunImpl();
    template <typename T> void PoolRunImpl();
    template <typename T> void PowRunImpl();
    template <typename T> void ProdRunImpl();

 protected:
    string mode_;
    int local_size_;
    float alpha_, beta_, k_;
    unique_ptr<OperatorBase> sqr_op_, pool_op_;
    unique_ptr<OperatorBase> pow_op_, prod_op_;
    Tensor *scale_, *sqr_in_, *prod_in_;
    Tensor *sqr_out_, *pool_out_, *pow_out_;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNLRNOp final : public LRNOp<Context> {
 public:
    CuDNNLRNOp(const OperatorDef& def, Workspace* ws)
        : LRNOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
        CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrn_desc_));
        CUDNN_CHECK(cudnnSetLRNDescriptor(
            lrn_desc_,
            this->local_size_,
            this->alpha_,
            this->beta_,
            this->k_
        ));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNLRNOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc_));
        CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrn_desc_));
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnLRNDescriptor_t lrn_desc_;
};

template <class Context>
class CuDNNLRNGradientOp final : public LRNGradientOp<Context > {
 public:
    CuDNNLRNGradientOp(const OperatorDef& def, Workspace* ws) 
        : LRNGradientOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
        CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrn_desc_));
        CUDNN_CHECK(cudnnSetLRNDescriptor(
            lrn_desc_,
            this->local_size_,
            this->alpha_,
            this->beta_,
            this->k_
        ));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNLRNGradientOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc_));
        CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrn_desc_));
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnLRNDescriptor_t lrn_desc_;
};

#endif  // WITH CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_LRN_OP_H_