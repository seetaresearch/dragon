// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_SIGMOID_OP_HPP
#define DRAGON_OPERATORS_ACTIVATION_SIGMOID_OP_HPP

#include "core/operator.h"

namespace dragon {

template <class Context>
class SigmoidOp : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(SigmoidOp);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class SigmoidGradientOp : public Operator<Context> {
 public:
    SigmoidGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {
        DISABLE_SHARE_GRADIENT;
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNSigmoidOp final : public SigmoidOp<Context> {
public:
    CuDNNSigmoidOp(const OperatorDef& op_def, Workspace* ws) 
        : SigmoidOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, 
            CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0));
    }

    ~CuDNNSigmoidOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType(); 

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t act_desc;
};

template <class Context>
class CuDNNSigmoidGradientOp final : public SigmoidGradientOp<Context> {
 public:
    CuDNNSigmoidGradientOp(const OperatorDef& op_def, Workspace* ws)
        : SigmoidGradientOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc,
            CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0));
    }

    ~CuDNNSigmoidGradientOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t act_desc;
};

#endif // WITH_CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_SIGMOID_OP_HPP