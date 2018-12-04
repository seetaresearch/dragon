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

#ifndef DRAGON_OPERATORS_RECURRENT_CUDNN_RECURRENT_OP_H_
#define DRAGON_OPERATORS_RECURRENT_CUDNN_RECURRENT_OP_H_

#include "core/operator.h"

namespace dragon {

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

class cudnnTensorDescriptors {
 public:
    cudnnTensorDescriptors(const int num_descs) {
        descs_.resize(num_descs);
        for (int i = 0; i < num_descs; ++i)
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs_[i]));
    }
    ~cudnnTensorDescriptors() {
        for (auto desc : descs_)
            cudnnDestroyTensorDescriptor(desc);
    }

    template <typename T>
    void Set(const vector<TIndex>& dims, const vector<TIndex>& strides) {
        CHECK_EQ(dims.size(), strides.size());
        for (auto desc : descs_) cudnnSetTensorDesc<T>(&desc, dims, strides);
    }

    const cudnnTensorDescriptor_t* descs() const { return descs_.data(); }

 protected:
    vector<cudnnTensorDescriptor_t> descs_;
};

template <class Context>
class CuDNNRecurrentOpBase : public Operator<Context> {
 public:
    CuDNNRecurrentOpBase(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws), states_initialized(false), 
          hidden_size(OperatorBase::Arg<int>("hidden_size", 0)),
          num_layers(OperatorBase::Arg<int>("num_layers", 1)),
          bidirectional(OperatorBase::Arg<bool>("bidirectional", false)),
          dropout_ratio(OperatorBase::Arg<float>("dropout_ratio", 1.f)),
          random_seed(def.device_option().random_seed()) {
        //  determine the rnn direction
        rnn_direction = bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
        //  determine the rnn mode
        const string mode = OperatorBase::Arg<string>("rnn_mode", "");
        if (mode == "rnn_tanh") rnn_mode = CUDNN_RNN_TANH;
        else if (mode == "rnn_relu") rnn_mode = CUDNN_RNN_RELU;
        else if (mode == "lstm") rnn_mode = CUDNN_LSTM;
        else if (mode == "gru") rnn_mode = CUDNN_GRU;
        else LOG(FATAL) << "Unsupported rnn mode: " << mode;
        //  determine the rnn input mode
        const string input_mode = OperatorBase::Arg<string>("rnn_input_mode", "linear");
        if (input_mode == "skip") rnn_input_mode = CUDNN_SKIP_INPUT;
        else if (input_mode == "linear") rnn_input_mode = CUDNN_LINEAR_INPUT;
        else LOG(FATAL) << "Unsupported rnn input mode: " << input_mode;
        //  override the running phase
        SwitchToPhase(OperatorBase::Arg<string>("phase", ""));
        CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc));
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&hx_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&cx_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&hy_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&cy_desc));
    }
    USE_OPERATOR_FUNCTIONS;

    virtual ~CuDNNRecurrentOpBase() {
        CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc));
        CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(hx_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(cx_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(hy_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(cy_desc));
    }

    template <typename T> void ResetDesc();

 public:
    TIndex hidden_size, num_layers;
    bool bidirectional, states_initialized;
    float dropout_ratio;
    unsigned long long random_seed;

    cudnnRNNDescriptor_t rnn_desc;
    cudnnDropoutDescriptor_t dropout_desc;
    cudnnDirectionMode_t rnn_direction;
    cudnnRNNMode_t rnn_mode;
    cudnnRNNInputMode_t rnn_input_mode;
    cudnnFilterDescriptor_t w_desc;
    cudnnTensorDescriptor_t hx_desc, cx_desc;
    cudnnTensorDescriptor_t hy_desc, cy_desc;
    vector<TIndex> input_dims, output_dims, hidden_dims;
    size_t workspace_size, reserve_size, states_size;

    std::unique_ptr<cudnnTensorDescriptors> xs_desc;
    std::unique_ptr<cudnnTensorDescriptors> ys_desc;
};

#define USE_CUDNN_RECURRENT_FUNCTIONS \
    USE_OPERATOR_FUNCTIONS; \
    using CuDNNRecurrentOpBase<Context>::dropout_desc; \
    using CuDNNRecurrentOpBase<Context>::rnn_desc; \
    using CuDNNRecurrentOpBase<Context>::w_desc; \
    using CuDNNRecurrentOpBase<Context>::hx_desc; \
    using CuDNNRecurrentOpBase<Context>::cx_desc; \
    using CuDNNRecurrentOpBase<Context>::hy_desc; \
    using CuDNNRecurrentOpBase<Context>::cy_desc; \
    using CuDNNRecurrentOpBase<Context>::xs_desc; \
    using CuDNNRecurrentOpBase<Context>::ys_desc; \
    using CuDNNRecurrentOpBase<Context>::input_dims; \
    using CuDNNRecurrentOpBase<Context>::output_dims; \
    using CuDNNRecurrentOpBase<Context>::hidden_dims; \
    using CuDNNRecurrentOpBase<Context>::workspace_size; \
    using CuDNNRecurrentOpBase<Context>::reserve_size

template <class Context>
class CuDNNRecurrentOp final : public CuDNNRecurrentOpBase<Context> {
 public:
    CuDNNRecurrentOp(const OperatorDef& def, Workspace* ws)
        : CuDNNRecurrentOpBase<Context>(def, ws) {}
    USE_CUDNN_RECURRENT_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class CuDNNRecurrentGradientOp final : public CuDNNRecurrentOpBase<Context> {
 public:
    CuDNNRecurrentGradientOp(const OperatorDef& def, Workspace* ws)
        : CuDNNRecurrentOpBase<Context>(def, ws) {}
    USE_CUDNN_RECURRENT_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

#endif

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_RECURRENT_CUDNN_RECURRENT_OP_H_