// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SoftmaxOp final : public Operator<Context> {
 public:
    SoftmaxOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType(); 

 protected:
    int axis;
    TIndex outer_dim, inner_dim;
    Tensor* sum_multiplier, *scale;
};

template <class Context>
class SoftmaxGradientOp final : public Operator<Context> {
 public:
    SoftmaxGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)) {
        DISABLE_SHARE_GRADIENT;
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int axis;
    TIndex outer_dim, inner_dim;
    Tensor* sum_multiplier, *scale;
};

#ifdef WITH_CUDNN

#include "utils/cudnn_device.h"

template <class Context>
class CuDNNSoftmaxOp final : public Operator<Context> {
 public:
    CuDNNSoftmaxOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    }

    ~CuDNNSoftmaxOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int axis;
    TIndex outer_dim, inner_dim;
    cudnnTensorDescriptor_t input_desc, output_desc;
};

template <class Context>
class CuDNNSoftmaxGradientOp final : public Operator<Context> {
 public:
    CuDNNSoftmaxGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    }

    ~CuDNNSoftmaxGradientOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int axis;
    TIndex outer_dim, inner_dim;
    cudnnTensorDescriptor_t input_desc, output_desc;
};

#endif    // WITH_CUDNN

}

#endif    // DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_