// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_LRN_OP_H_
#define DRAGON_OPERATORS_VISION_LRN_OP_H_

#include "core/operator.h"

namespace dragon {

enum LRNMode { ACROSS_CHANNELS, WITHIN_CHANNEL };

template <class Context>
class LRNOp : public Operator<Context> {
 public:
    LRNOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          local_size(OperatorBase::GetSingleArg<int>("local_size", 5)),
          alpha(OperatorBase::GetSingleArg<float>("alpha", float(0.0001))),
          beta(OperatorBase::GetSingleArg<float>("beta", float(0.75))),
          k(OperatorBase::GetSingleArg<float>("k", float(2.0))),
          mode(OperatorBase::GetSingleArg<string>("mode", "ACROSS_CHANNELS")),
          data_format(OperatorBase::GetSingleArg<string>("data_format", "NCHW")) {}

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
    LRNGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          local_size(OperatorBase::GetSingleArg<int>("local_size", 5)),
          alpha(OperatorBase::GetSingleArg<float>("alpha", float(0.0001))),
          beta(OperatorBase::GetSingleArg<float>("beta", float(0.75))),
          k(OperatorBase::GetSingleArg<float>("k", float(2.0))),
          mode(OperatorBase::GetSingleArg<string>("mode", "ACROSS_CHANNELS")),
          data_format(OperatorBase::GetSingleArg<string>("data_format", "NCHW")) {}

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

#include "utils/cudnn_device.h"

template <class Context>
class CuDNNLRNOp : public LRNOp<Context> {
 public:
    CuDNNLRNOp(const OperatorDef& op_def, Workspace* ws) 
        : LRNOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc));
        CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc, this->local_size, 
                                                     this->alpha, 
                                                     this->beta, 
                                                     this->k));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnLRNDescriptor_t norm_desc;
};

template <class Context>
class CuDNNLRNGradientOp : public LRNGradientOp<Context > {
 public:
    CuDNNLRNGradientOp(const OperatorDef& op_def, Workspace* ws) :
        LRNGradientOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc));
        CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc, this->local_size,
                                                     this->alpha, 
                                                     this->beta, 
                                                     this->k));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnLRNDescriptor_t norm_desc;
};

#endif    // WITH CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_LRN_OP_H_