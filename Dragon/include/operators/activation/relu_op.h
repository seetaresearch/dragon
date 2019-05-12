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

#ifndef DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ReluOp : public Operator<Context> {
 public:
    ReluOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          slope_(OpArg<float>("slope", 0.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float slope_;
};

template <class Context>
class ReluGradientOp : public Operator<Context> {
 public:
    ReluGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          slope_(OpArg<float>("slope", 0.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float slope_;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNReluOp final : public ReluOp<Context> {
public:
    CuDNNReluOp(const OperatorDef& def, Workspace* ws)
        : ReluOp<Context>(def, ws) {
        CuDNNCreateTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            act_desc_,
            CUDNN_ACTIVATION_RELU,
            CUDNN_PROPAGATE_NAN, 0
        ));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNReluOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc_));
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    cudnnTensorDescriptor_t input_desc_;
    cudnnActivationDescriptor_t act_desc_;
};

template <class Context>
class CuDNNReluGradientOp final : public ReluGradientOp<Context> {
 public:
    CuDNNReluGradientOp(const OperatorDef& def, Workspace* ws)
        : ReluGradientOp<Context>(def, ws) {
        CuDNNCreateTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            act_desc_,
            CUDNN_ACTIVATION_RELU,
            CUDNN_PROPAGATE_NAN, 0
        ));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNReluGradientOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc_));
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    cudnnTensorDescriptor_t input_desc_;
    cudnnActivationDescriptor_t act_desc_;
};

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_