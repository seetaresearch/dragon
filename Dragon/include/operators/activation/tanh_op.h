// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_TANH_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_TANH_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class TanhOp : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(TanhOp);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class TanhGradientOp : public Operator<Context> {
 public:
     TanhGradientOp(const OperatorDef& op_def, Workspace* ws)
         : Operator<Context>(op_def, ws) {
         DISABLE_SHARE_GRADIENT;
     }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNTanhOp final : public TanhOp<Context> {
public:
    CuDNNTanhOp(const OperatorDef& op_def, Workspace* ws)
        : TanhOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, 
            CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));
    }
    void RunOnDevice() override;
    template <typename T> void RunWithType(); 

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t act_desc;
};

template <class Context>
class CuDNNTanhGradientOp final : public TanhGradientOp<Context> {
 public:
    CuDNNTanhGradientOp(const OperatorDef& op_def, Workspace* ws)
        : TanhGradientOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc,
            CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));
    }
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t act_desc;
};

#endif // WITH_CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_TANH_OP_H_