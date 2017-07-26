// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ReluOp : public Operator<Context> {
 public:
    ReluOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          slope(OperatorBase::GetSingleArg<float>("slope", 0.0)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float slope;
};

template <class Context>
class ReluGradientOp : public Operator<Context> {
 public:
    ReluGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          slope(OperatorBase::GetSingleArg<float>("slope", 0.0)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float slope;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNReluOp final : public ReluOp<Context> {
public:
    CuDNNReluOp(const OperatorDef& op_def, Workspace* ws) 
        : ReluOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, 
            CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
    }
    void RunOnDevice() override;
    template <typename T> void RunWithType(); 

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t act_desc;
};

template <class Context>
class CuDNNReluGradientOp final : public ReluGradientOp<Context> {
 public:
    CuDNNReluGradientOp(const OperatorDef& op_def, Workspace* ws)
        : ReluGradientOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc,
            CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
    }
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t act_desc;
};

#endif // WITH_CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_