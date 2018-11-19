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

#ifndef DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ReluOp : public Operator<Context> {
 public:
    ReluOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          slope(OperatorBase::Arg<float>("slope", 0.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float slope;
};

template <class Context>
class ReluGradientOp : public Operator<Context> {
 public:
    ReluGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          slope(OperatorBase::Arg<float>("slope", 0.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float slope;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNReluOp final : public ReluOp<Context> {
public:
    CuDNNReluOp(const OperatorDef& def, Workspace* ws)
        : ReluOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, 
            CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNReluOp() {
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
class CuDNNReluGradientOp final : public ReluGradientOp<Context> {
 public:
    CuDNNReluGradientOp(const OperatorDef& def, Workspace* ws)
        : ReluGradientOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc,
            CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNReluGradientOp() {
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

#endif    // DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_