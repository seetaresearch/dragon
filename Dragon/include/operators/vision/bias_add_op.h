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

#ifndef DRAGON_OPERATORS_VISION_BIAS_ADD_OP_H_
#define DRAGON_OPERATORS_VISION_BIAS_ADD_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class BiasAddOp final : public Operator<Context> {
 public:
    BiasAddOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          data_format(OperatorBase::Arg<string>(
              "data_format", "NCHW")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex outer_dim, dim, inner_dim;
    string data_format;
};

template <class Context>
class BiasAddGradientOp final : public Operator<Context> {
 public:
    BiasAddGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          data_format(OperatorBase::Arg<string>(
              "data_format", "NCHW")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex outer_dim, dim, inner_dim;
    string data_format;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNBiasAddOp final : public Operator<Context> {
 public:
    CuDNNBiasAddOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          data_format(OperatorBase::Arg<string>(
              "data_format", "NCHW")) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNBiasAddOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex outer_dim, dim, inner_dim;
    string data_format;
    cudnnTensorDescriptor_t bias_desc, output_desc;
};

template <class Context>
class CuDNNBiasAddGradientOp final : public Operator<Context> {
public:
    CuDNNBiasAddGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          data_format(OperatorBase::Arg<string>(
              "data_format", "NCHW")) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNBiasAddGradientOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

protected:
    TIndex outer_dim, dim, inner_dim;
    string data_format;
    cudnnTensorDescriptor_t input_desc, bias_desc;
};

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_BIAS_ADD_OP_H_