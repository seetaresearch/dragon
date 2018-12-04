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

#ifndef DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class EluOp : public Operator<Context> {
 public:
    EluOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          alpha(OperatorBase::Arg<float>("alpha", 1.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float alpha;
};

template <class Context>
class EluGradientOp : public Operator<Context> {
 public:
    EluGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          alpha(OperatorBase::Arg<float>("alpha", 1.f)) {}
    USE_OPERATOR_FUNCTIONS;

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
    CuDNNEluOp(const OperatorDef& def, Workspace* ws)
        : EluOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc,
            CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, this->alpha));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNEluOp() {
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
class CuDNNEluGradientOp final : public EluGradientOp<Context> {
 public:
    CuDNNEluGradientOp(const OperatorDef& def, Workspace* ws)
        : EluGradientOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc,
            CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, this->alpha));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNEluGradientOp() {
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

#endif

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_