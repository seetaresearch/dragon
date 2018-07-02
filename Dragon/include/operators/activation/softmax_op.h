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

#ifndef DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SoftmaxOp final : public Operator<Context> {
 public:
    SoftmaxOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType(); 

 protected:
    TIndex axis, outer_dim, inner_dim;
};

template <class Context>
class SoftmaxGradientOp final : public Operator<Context> {
 public:
    SoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim;
};

#ifdef WITH_CUDNN

#include "utils/cudnn_device.h"

template <class Context>
class CuDNNSoftmaxOp final : public Operator<Context> {
 public:
    CuDNNSoftmaxOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 1)) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    }
    USE_OPERATOR_FUNCTIONS;

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
    CuDNNSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 1)) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    }
    USE_OPERATOR_FUNCTIONS;

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