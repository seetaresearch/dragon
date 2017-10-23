// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class EluOp : public Operator<Context> {
 public:
    EluOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          alpha(OperatorBase::GetSingleArg<float>("alpha", 1.0)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float alpha;
};

template <class Context>
class EluGradientOp : public Operator<Context> {
 public:
    EluGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          alpha(OperatorBase::GetSingleArg<float>("alpha", 1.0)) {
        DISABLE_SHARE_GRADIENT;
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float alpha;
};

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(6, 0, 0)

template <class Context>
class CuDNNEluOp final : public EluOp<Context> {
public:
    CuDNNEluOp(const OperatorDef& op_def, Workspace* ws) 
        : EluOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, 
            CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, this->alpha));
    }
    void RunOnDevice() override;
    template <typename T> void RunWithType(); 

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t act_desc;
};

template <class Context>
class CuDNNEluGradientOp final : public EluGradientOp<Context> {
 public:
    CuDNNEluGradientOp(const OperatorDef& op_def, Workspace* ws)
        : EluGradientOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc,
            CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, this->alpha));
    }
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t act_desc;
};

#endif

#endif // WITH_CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_