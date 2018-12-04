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

#ifndef DRAGON_OPERATORS_VISION_POOLING_OP_H_
#define DRAGON_OPERATORS_VISION_POOLING_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class Pooling2dOp : public Operator<Context> {
 public:
    Pooling2dOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          mode(OperatorBase::Arg<string>("mode", "MAX")),
          data_format(OperatorBase::Arg<string>("data_format", "NCHW")),
          padding(OperatorBase::Arg<string>("padding", "VALID")),
          global_pooling(OperatorBase::Arg<bool>("global_pooling", false)),
          ceil_mode(OperatorBase::Arg<bool>("ceil", true)) {
        vector<int> ks = OperatorBase::Args<int>("kernel_size");
        vector<int> s = OperatorBase::Args<int>("stride");
        vector<int> p = OperatorBase::Args<int>("pad");
        for (int i = 0; i < 2; i++) {
            if (global_pooling) {
                kernel_size.push_back(-1);
                stride.push_back(1);
                pad.push_back(0);
            } else {
                kernel_size.push_back(i < ks.size() ? ks[i] : ks[0]);
                stride.push_back(i < s.size() ? s[i] : s[0]);
                pad.push_back(i < p.size() ? p[i] : p[0]);
             }
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void Reshape();
    void RunOnDevice() override;
    template <typename T> void MAXRunWithType();
    template <typename T> void AVGRunWithType();

 protected:
    vector<TIndex> kernel_size, stride, pad;
    Tensor* mask;
    string mode, data_format, padding;
    TIndex n, c, h, w, pool_h, pool_w;
    bool global_pooling, ceil_mode;
};

template <class Context>
class Pooling2dGradientOp : public Operator<Context> {
 public:
    Pooling2dGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          mode(OperatorBase::Arg<string>("mode", "MAX")),
          data_format(OperatorBase::Arg<string>("data_format", "NCHW")),
          padding(OperatorBase::Arg<string>("padding", "VALID")),
          global_pooling(OperatorBase::Arg<bool>("global_pooling", false)),
          ceil_mode(OperatorBase::Arg<bool>("ceil", true)) {
        vector<int> ks = OperatorBase::Args<int>("kernel_size");
        vector<int> s = OperatorBase::Args<int>("stride");
        vector<int> p = OperatorBase::Args<int>("pad");
        for (int i = 0; i < 2; i++) {
            if (global_pooling) {
                kernel_size.push_back(-1);
                stride.push_back(1);
                pad.push_back(0);
            } else {
                kernel_size.push_back(i < ks.size() ? ks[i] : ks[0]);
                stride.push_back(i < s.size() ? s[i] : s[0]);
                pad.push_back(i < p.size() ? p[i] : p[0]);
            }
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void Reshape();
    void RunOnDevice() override;
    template <typename T> void MAXRunWithType();
    template <typename T> void AVGRunWithType();

 protected:
    vector<TIndex> kernel_size, stride, pad;
    Tensor* mask;
    string mode, data_format, padding;
    TIndex n, c, h, w, pool_h, pool_w;
    bool global_pooling, ceil_mode;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNPooling2dOp final : public Pooling2dOp<Context> {
 public:
    CuDNNPooling2dOp(const OperatorDef& def, Workspace* ws)
        : Pooling2dOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
        if (this->mode == "MAX") {
#if CUDNN_VERSION_MIN(6,0,0)
            pool_mode = CUDNN_POOLING_MAX_DETERMINISTIC;
#else
            pool_mode = CUDNN_POOLING_MAX;
#endif
        } else if (this->mode == "AVG") {
            pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        } else LOG(FATAL) << "Unsupported pooling mode: " << this->mode;
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNPooling2dOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t pool_desc;
    cudnnPoolingMode_t pool_mode;
};

template <class Context>
class CuDNNPooling2dGradientOp final : public Pooling2dGradientOp<Context> {
 public:
    CuDNNPooling2dGradientOp(const OperatorDef& def, Workspace* ws)
        : Pooling2dGradientOp<Context>(def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
        if (this->mode == "MAX") {
#if CUDNN_VERSION_MIN(6,0,0)
            pool_mode = CUDNN_POOLING_MAX_DETERMINISTIC;
#else
            pool_mode = CUDNN_POOLING_MAX;
#endif
        } else if (this->mode == "AVG") {
            pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        } else LOG(FATAL) << "Unsupported pooling mode: " << this->mode;
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNPooling2dGradientOp() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t pool_desc;
    cudnnPoolingMode_t pool_mode;
};

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_POOLING_OP_H_