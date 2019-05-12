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

#ifndef DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SoftmaxOp final : public Operator<Context> {
 public:
    SoftmaxOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t axis_, axis_dim_;
    int64_t outer_dim_, inner_dim_;
};

template <class Context>
class SoftmaxGradientOp final : public Operator<Context> {
 public:
    SoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t axis_, axis_dim_;
    int64_t outer_dim_, inner_dim_;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNSoftmaxOp final : public Operator<Context> {
 public:
    CuDNNSoftmaxOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)) {
        CuDNNCreateTensorDesc(&input_desc_);
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNSoftmaxOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t axis_, outer_dim_, inner_dim_;
    cudnnTensorDescriptor_t input_desc_;
};

template <class Context>
class CuDNNSoftmaxGradientOp final : public Operator<Context> {
 public:
    CuDNNSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)) {
        CuDNNCreateTensorDesc(&input_desc_);
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNSoftmaxGradientOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t axis_, outer_dim_, inner_dim_;
    cudnnTensorDescriptor_t input_desc_;
};

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_