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
          alpha_(OpArg<float>("alpha", 1.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float alpha_;
};

template <class Context>
class EluGradientOp : public Operator<Context> {
 public:
    EluGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          alpha_(OpArg<float>("alpha", 1.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float alpha_;
};

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(6, 0, 0)

template <class Context>
class CuDNNEluOp final : public EluOp<Context> {
public:
    CuDNNEluOp(const OperatorDef& def, Workspace* ws)
        : EluOp<Context>(def, ws) {
        CuDNNCreateTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            act_desc_,
            CUDNN_ACTIVATION_ELU,
            CUDNN_PROPAGATE_NAN,
            this->alpha_
        ));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNEluOp() {
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
class CuDNNEluGradientOp final : public EluGradientOp<Context> {
 public:
    CuDNNEluGradientOp(const OperatorDef& def, Workspace* ws)
        : EluGradientOp<Context>(def, ws) {
        CuDNNCreateTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            act_desc_,
            CUDNN_ACTIVATION_ELU,
            CUDNN_PROPAGATE_NAN,
            this->alpha_
        ));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNEluGradientOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc_));
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    cudnnTensorDescriptor_t input_desc_;
    cudnnActivationDescriptor_t act_desc_;
};

#endif

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_