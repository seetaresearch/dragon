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

#ifndef DRAGON_OPERATORS_VISION_POOL_OP_H_
#define DRAGON_OPERATORS_VISION_POOL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class Pool2dOp : public Operator<Context> {
 public:
    Pool2dOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          mode_(OpArg<string>("mode", "MAX")),
          padding_(OpArg<string>("padding", "VALID")),
          ceil_mode_(OpArg<bool>("ceil_mode", true)),
          global_pool_(OpArg<bool>("global_pooling", false)) {
        auto at = [&](const vec64_t & vec, int i) {
            return i < vec.size() ? vec[i] : vec[0];
        };
        auto pads = OpArgs<int64_t>("pads");
        auto strides = OpArgs<int64_t>("strides");
        auto kshape = OpArgs<int64_t>("kernel_shape");
        for (int i = 0; i < 2; i++) {
            if (global_pool_) {
                pad_l_.push_back(0);
                stride_.push_back(1);
                kshape_.push_back(-1);
            } else {
                pad_l_.push_back(at(pads, i));
                stride_.push_back(at(strides, i));
                kshape_.push_back(at(kshape, i));
             }
        }
        if (pads.size() == 4) {
            pad_r_.assign(pads.begin() + 2, pads.end());
        } else {
            pad_r_.assign(pad_l_.begin(), pad_l_.end());
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void Reshape();

    void RunOnDevice() override;
    template <typename T> void MaxRunImpl();
    template <typename T> void AvgRunImpl();

 protected:
    string mode_, padding_;
    bool global_pool_, ceil_mode_;
    vec64_t kshape_, stride_, pad_l_, pad_r_;
    int64_t n_, c_, h_, w_, pool_h_, pool_w_;
};

template <class Context>
class Pool2dGradientOp : public Operator<Context> {
 public:
    Pool2dGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          mode_(OpArg<string>("mode", "MAX")),
          padding_(OpArg<string>("padding", "VALID")),
          ceil_mode_(OpArg<bool>("ceil_mode", true)),
          global_pool_(OpArg<bool>("global_pooling", false)) {
        auto at = [&](const vec64_t & vec, int i) {
            return i < vec.size() ? vec[i] : vec[0];
        };
        auto pads = OpArgs<int64_t>("pads");
        auto strides = OpArgs<int64_t>("strides");
        auto kshape = OpArgs<int64_t>("kernel_shape");
        for (int i = 0; i < 2; i++) {
            if (global_pool_) {
                pad_l_.push_back(0);
                stride_.push_back(1);
                kshape_.push_back(-1);
            } else {
                pad_l_.push_back(at(pads, i));
                stride_.push_back(at(strides, i));
                kshape_.push_back(at(kshape, i));
             }
        }
        if (pads.size() == 4) {
            pad_r_.assign(pads.begin() + 2, pads.end());
        } else {
            pad_r_.assign(pad_l_.begin(), pad_l_.end());
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void Reshape();
    void RunOnDevice() override;
    template <typename T> void MaxRunImpl();
    template <typename T> void AvgRunImpl();

 protected:
    string mode_, padding_;
    bool global_pool_, ceil_mode_;
    vec64_t kshape_, stride_, pad_l_, pad_r_;
    int64_t n_, c_, h_, w_, pool_h_, pool_w_;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNPool2dOp final : public Pool2dOp<Context> {
 public:
    CuDNNPool2dOp(const OperatorDef& def, Workspace* ws)
        : Pool2dOp<Context>(def, ws) {
        CuDNNCreateTensorDesc(&input_desc_);
        CuDNNCreateTensorDesc(&output_desc_);
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
        if (this->mode_ == "MAX") {
#if CUDNN_VERSION_MIN(6,0,0)
            pool_mode_ = CUDNN_POOLING_MAX_DETERMINISTIC;
#else
            pool_mode_ = CUDNN_POOLING_MAX;
#endif
        } else if (this->mode_ == "AVG") {
            pool_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        } else {
            LOG(FATAL) << "Unknown Mode: " << this->mode_;
        }
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNPool2dOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CuDNNDestroyTensorDesc(&output_desc_);
        CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnPoolingDescriptor_t pool_desc_;
    cudnnPoolingMode_t pool_mode_;
};

template <class Context>
class CuDNNPool2dGradientOp final : public Pool2dGradientOp<Context> {
 public:
    CuDNNPool2dGradientOp(const OperatorDef& def, Workspace* ws)
        : Pool2dGradientOp<Context>(def, ws) {
        CuDNNCreateTensorDesc(&input_desc_);
        CuDNNCreateTensorDesc(&output_desc_);
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
        if (this->mode_ == "MAX") {
#if CUDNN_VERSION_MIN(6,0,0)
            pool_mode_ = CUDNN_POOLING_MAX_DETERMINISTIC;
#else
            pool_mode_ = CUDNN_POOLING_MAX;
#endif
        } else if (this->mode_ == "AVG") {
            pool_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        } else {
            LOG(FATAL) << "Unknown Mode: " << this->mode_;
        }
    }
    USE_OPERATOR_FUNCTIONS;

    ~CuDNNPool2dGradientOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CuDNNDestroyTensorDesc(&output_desc_);
        CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnPoolingDescriptor_t pool_desc_;
    cudnnPoolingMode_t pool_mode_;
};

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_POOLING_OP_H_