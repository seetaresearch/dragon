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
          use_scale_(OpArg<bool>("scale", true)) {
        SwitchToPhase(OpArg<string>("phase", ""));
        GET_ARG_WITH_DESC(float, prob, 0.5f);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    bool use_scale_;
    DECLARE_ARG_WITH_DESC(float, prob);
};

template <class Context>
class DropoutGradientOp final : public Operator<Context> {
 public:
    DropoutGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          use_scale_(OpArg<bool>("scale", true)) {
        SwitchToPhase(OpArg<string>("phase", ""));
        GET_ARG_WITH_DESC(float, prob, 0.5f);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    bool use_scale_;
    DECLARE_ARG_WITH_DESC(float, prob);
};

DEFINE_ARG_WITH_DESC(float, DropoutOp, prob);
DEFINE_ARG_WITH_DESC(float, DropoutGradientOp, prob);

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(7, 0, 0)

template <class Context>
class CuDNNDropoutOp final : public Operator<Context> {
 public:
    CuDNNDropoutOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          states_initialized_(false),
          rng_seed_(DEFAULT_RNG_SEED),
          use_scale_(OpArg<bool>("scale", true)) {
        SwitchToPhase(OpArg<string>("phase", ""));
        GET_ARG_WITH_DESC(float, prob, 0.5f);
        CuDNNCreateTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNDropoutOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    bool use_scale_, states_initialized_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnDropoutDescriptor_t dropout_desc_;
    size_t states_size_, reserve_size_;
    unsigned long long rng_seed_;
    DECLARE_ARG_WITH_DESC(float, prob);
};

template <class Context>
class CuDNNDropoutGradientOp final : public Operator<Context> {
public:
    CuDNNDropoutGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          states_initialized_(false),
          rng_seed_(DEFAULT_RNG_SEED),
          use_scale_(OpArg<bool>("scale", true)) {
        SwitchToPhase(OpArg<string>("phase", ""));
        GET_ARG_WITH_DESC(float, prob, 0.5f);
        CuDNNCreateTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNDropoutGradientOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    bool use_scale_, states_initialized_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnDropoutDescriptor_t dropout_desc_;
    size_t states_size_, reserve_size_;
    unsigned long long rng_seed_;
    DECLARE_ARG_WITH_DESC(float, prob);
};

DEFINE_ARG_WITH_DESC(float, CuDNNDropoutOp, prob);
DEFINE_ARG_WITH_DESC(float, CuDNNDropoutGradientOp, prob);

#endif

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_