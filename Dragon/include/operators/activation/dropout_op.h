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

#ifndef DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_

#include "core/operator.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
class DropoutOp final : public Operator<Context> {
 public:
    DropoutOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          use_scale(OperatorBase::Arg<bool>("scale", true)) {
        GET_ARGUMENT_WITH_DESC(float, prob, 0.5);
        SwitchToPhase(OperatorBase::Arg<string>("phase", ""));
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    DECLARE_ARGUMENT_WITH_DESC(float, prob);
    bool use_scale;
};

template <class Context>
class DropoutGradientOp final : public Operator<Context> {
 public:
    DropoutGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          use_scale(OperatorBase::Arg<bool>("scale", true)) {
        GET_ARGUMENT_WITH_DESC(float, prob, 0.5);
        SwitchToPhase(OperatorBase::Arg<string>("phase", ""));
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
     DECLARE_ARGUMENT_WITH_DESC(float, prob);
     bool use_scale;
     Tensor* mask;
};

DEFINE_ARGUMENT_WITH_DESC(float, DropoutOp, prob);
DEFINE_ARGUMENT_WITH_DESC(float, DropoutGradientOp, prob);

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(7, 0, 0)

template <class Context>
class CuDNNDropoutOp final : public Operator<Context> {
public:
    CuDNNDropoutOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws), states_initialized(false),
        use_scale(OperatorBase::Arg<bool>("scale", true)),
        random_seed(DEFAULT_RNG_SEED) {
        GET_ARGUMENT_WITH_DESC(float, prob, 0.5);
        SwitchToPhase(OperatorBase::Arg<string>("phase", ""));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNDropoutOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType(); 

 protected:
    DECLARE_ARGUMENT_WITH_DESC(float, prob);
    bool use_scale, states_initialized;
    cudnnTensorDescriptor_t input_desc;
    cudnnDropoutDescriptor_t dropout_desc;
    size_t states_size, reserve_space_size;
    unsigned long long random_seed;
};

template <class Context>
class CuDNNDropoutGradientOp final : public Operator<Context> {
public:
    CuDNNDropoutGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws), states_initialized(false),
        use_scale(OperatorBase::Arg<bool>("scale", true)),
        random_seed(DEFAULT_RNG_SEED) {
        GET_ARGUMENT_WITH_DESC(float, prob, 0.5);
        SwitchToPhase(OperatorBase::Arg<string>("phase", ""));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNDropoutGradientOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType(); 

 protected:
    DECLARE_ARGUMENT_WITH_DESC(float, prob);
    bool use_scale, states_initialized;
    cudnnTensorDescriptor_t input_desc;
    cudnnDropoutDescriptor_t dropout_desc;
    size_t states_size, reserve_space_size;
    unsigned long long random_seed;
};

DEFINE_ARGUMENT_WITH_DESC(float, CuDNNDropoutOp, prob);
DEFINE_ARGUMENT_WITH_DESC(float, CuDNNDropoutGradientOp, prob);

#endif

#endif // WITH_CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_