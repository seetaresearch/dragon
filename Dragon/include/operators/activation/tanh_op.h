// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_TANH_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_TANH_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class TanhOp : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(TanhOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class TanhGradientOp : public Operator<Context> {
 public:
     USE_SIMPLE_CTOR_DTOR(TanhGradientOp);
     USE_OPERATOR_FUNCTIONS;

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
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNTanhOp() {
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
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNTanhGradientOp() {
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

#endif    // DRAGON_OPERATORS_ACTIVATION_TANH_OP_H_