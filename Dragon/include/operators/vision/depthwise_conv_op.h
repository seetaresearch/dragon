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

#ifndef DRAGON_OPERATORS_VISION_DEPTHWISE_CONV_OP_H_
#define DRAGON_OPERATORS_VISION_DEPTHWISE_CONV_OP_H_

#include "operators/vision/conv_op_base.h"

namespace dragon {

template <class Context>
class DepthwiseConv2dOp : public ConvOpBase<Context> {
 public:
    DepthwiseConv2dOp(const OperatorDef& def, Workspace* ws)
        : ConvOpBase<Context>(def, ws) {
        this->num_spatial_axes = 2;
        Setup();
        CHECK_EQ(stride[0], stride[1])
            << "Excepted stride_h == stride_w";
    }
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS;

    bool ReverseDimensions() override { return false; }
    bool HasBias() override { return InputSize() > 2; }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
};

template <class Context>
class DepthwiseConv2dGradientOp
    : public DepthwiseConv2dOp<Context> {
 public:
    DepthwiseConv2dGradientOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : DepthwiseConv2dOp<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS;

    bool HasBias() override { return Output(2)->name() != "NULL"; }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNDepthwiseConv2dOp final
    : public DepthwiseConv2dOp<Context> {
 public:
     CuDNNDepthwiseConv2dOp(
         const OperatorDef&         def,
         Workspace*                 ws)
        : DepthwiseConv2dOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    }
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS;

    ~CuDNNDepthwiseConv2dOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t bias_desc, output_desc;
};

template <class Context>
class CuDNNDepthwiseConv2dGradientOp final
    : public DepthwiseConv2dGradientOp<Context> {
 public:
     CuDNNDepthwiseConv2dGradientOp(
         const OperatorDef&         def,
         Workspace*                 ws)
        : DepthwiseConv2dGradientOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    }
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS;

    ~CuDNNDepthwiseConv2dGradientOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t bias_desc, input_desc;
};

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_DEPTHWISE_CONV_OP_H_