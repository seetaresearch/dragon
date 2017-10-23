// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_POOLING_OP_H_
#define DRAGON_OPERATORS_VISION_POOLING_OP_H_

#include "core/operator.h"

namespace dragon {

enum PoolingMode { MAX_POOLING, AVG_POOLING };

template <class Context>
class PoolingOp: public Operator <Context> {
 public:
     PoolingOp(const OperatorDef& op_def, Workspace* ws)
         : Operator<Context>(op_def, ws),
           mode(PoolingMode(OperatorBase::GetSingleArg<int>("mode", MAX_POOLING))),
           global_pooling(OperatorBase::GetSingleArg<bool>("global_pooling", false)) {
         vector<int> ks = OperatorBase::GetRepeatedArg<int>("kernel_size");
         vector<int> s = OperatorBase::GetRepeatedArg<int>("stride");
         vector<int> p = OperatorBase::GetRepeatedArg<int>("pad");
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

    void Reshape();
    void RunOnDevice() override;
    template <typename T> void MaxRunWithType();
    template <typename T> void AvgRunWithType();

 protected:
    vector<TIndex> kernel_size, stride, pad;
    Tensor* mask;
    PoolingMode mode;
    TIndex num, channels, height, width;
    TIndex pool_height, pool_width;
    bool global_pooling;
};

template <class Context>
class PoolingGradientOp: public Operator<Context> {
 public:
    PoolingGradientOp(const OperatorDef& op_def, Workspace* ws)
         : Operator<Context>(op_def, ws),
           mode(PoolingMode(OperatorBase::GetSingleArg<int>("mode", MAX_POOLING))),
           global_pooling(OperatorBase::GetSingleArg<bool>("global_pooling", false)) {
         vector<int> ks = OperatorBase::GetRepeatedArg<int>("kernel_size");
         vector<int> s = OperatorBase::GetRepeatedArg<int>("stride");
         vector<int> p = OperatorBase::GetRepeatedArg<int>("pad");
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

    void Reshape();
    void RunOnDevice() override;
    template <typename T> void MaxRunWithType();
    template <typename T> void AvgRunWithType();

 protected:
    vector<TIndex> kernel_size, stride, pad;
    Tensor* mask;
    PoolingMode mode;
    TIndex num, channels, height, width;
    TIndex pool_height, pool_width;
    bool global_pooling;
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNPoolingOp final : public PoolingOp<Context> {
 public:
    CuDNNPoolingOp(const OperatorDef& op_def, Workspace* ws)
        : PoolingOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
        pool_mode = this->mode == MAX_POOLING ? 
                                  CUDNN_POOLING_MAX :
                                  CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
#if CUDNN_VERSION_MIN(5, 0, 0)
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc, 
                                                pool_mode,
                                      CUDNN_PROPAGATE_NAN, 
               this->kernel_size[0], this->kernel_size[1],
                               this->pad[0], this->pad[1], 
                       this->stride[0], this->stride[1]));
#else
        CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(pool_desc, 
                                                   pool_mode,
                                         CUDNN_PROPAGATE_NAN, 
                  this->kernel_size[0], this->kernel_size[1],
                                  this->pad[0], this->pad[1], 
                          this->stride[0], this->stride[1]));
#endif
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t  pool_desc;
    cudnnPoolingMode_t        pool_mode;
};

template <class Context>
class CuDNNPoolingGradientOp final : public PoolingGradientOp<Context> {
 public:
    CuDNNPoolingGradientOp(const OperatorDef& op_def, Workspace* ws)
        : PoolingGradientOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
        pool_mode = this->mode == MAX_POOLING ? 
                                  CUDNN_POOLING_MAX :
                                  CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
#if CUDNN_VERSION_MIN(5, 0, 0)
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc, 
                                                pool_mode,
                                      CUDNN_PROPAGATE_NAN, 
               this->kernel_size[0], this->kernel_size[1],
                               this->pad[0], this->pad[1], 
                       this->stride[0], this->stride[1]));
#else
        CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(pool_desc, 
                                                   pool_mode,
                                         CUDNN_PROPAGATE_NAN, 
                  this->kernel_size[0], this->kernel_size[1],
                                  this->pad[0], this->pad[1], 
                          this->stride[0], this->stride[1]));
#endif
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t  pool_desc;
    cudnnPoolingMode_t        pool_mode;
};

#endif    // WITH_CUDNN

}    // namespace dragon
    
#endif    // DRAGON_OPERATORS_VISION_POOLING_OP_H_