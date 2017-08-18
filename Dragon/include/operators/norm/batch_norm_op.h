// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_
#define DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class BatchNormOp : public Operator<Context> {
 public:
    BatchNormOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          momentum(OperatorBase::GetSingleArg<float>("momentum", float(0.9))),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)),
          inplace(OperatorBase::GetSingleArg<bool>("inplace", false)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float momentum, eps;
    Tensor mean, num_by_chans;
    Tensor* num_multiplier, *spatial_multiplier, *stddev, *var;
    TIndex num, channels, spatial_dim, nbychans;
    int use_stats;
    bool use_global_stats, inplace, is_recomputing;
};

template <class Context>
class BatchNormGradientOp final : public Operator<Context> {
 public:
    BatchNormGradientOp(const OperatorDef& op_def, Workspace *ws)
        : Operator<Context>(op_def, ws),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor num_by_chans;
    Tensor* num_multiplier, *spatial_multiplier, *stddev, *var;
    TIndex num, channels, spatial_dim, nbychans;
    int use_stats;
    bool use_global_stats;
};

template <class Context>
class BNOp : public Operator<Context> {
 public:
    BNOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          momentum(OperatorBase::GetSingleArg<float>("momentum", float(0.9))),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) { }

    void RunOnDevice() override { NOT_IMPLEMENTED; }
    template <typename T> void RunWithType() { NOT_IMPLEMENTED; }
  
 protected:
    float momentum, eps;
    int use_stats;
    bool use_global_stats, is_recomputing;
};

template <class Context>
class BNGradientOp : public Operator<Context> {
 public:
    BNGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) { }

    void ShareGradient() override;
    void RunOnDevice() override { NOT_IMPLEMENTED; }
    template <typename T> void RunWithType() { NOT_IMPLEMENTED; }
  
 protected:
    float eps;
    int use_stats;
    bool use_global_stats;
};

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

#include "utils/cudnn_device.h"

template <class Context>
class CuDNNBNOp final : public BNOp<Context> {
 public:
    CuDNNBNOp(const OperatorDef& op_def, Workspace* ws)
        : BNOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
        this->eps = std::max(this->eps, float(CUDNN_BN_MIN_EPSILON));
    }

    void RunOnDevice() override;
    template <typename T> void SpatialRunWithType();
    template <typename T> void PerActivationRunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc, bn_desc;
    TIndex num, channels, spatial_dim;
    Tensor* mean, *var;
    bool use_global_stats, is_recomputing;
};

template <class Context>
class CuDNNBNGradientOp final : public BNGradientOp<Context> {
 public:
    CuDNNBNGradientOp(const OperatorDef& op_def, Workspace* ws)
        : BNGradientOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
        this->eps = std::max(this->eps, float(CUDNN_BN_MIN_EPSILON));
    }

    void RunOnDevice() override;
    template <typename T> void SpatialRunWithType();
    template <typename T> void PerActivationRunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc, bn_desc;
    Tensor num_by_chans;
    Tensor* num_multiplier, *spatial_multiplier;
    Tensor* mean, *var, *stddev;
    TIndex num, channels, spatial_dim, nbychans;
    bool use_global_stats;
};

#endif

#endif  // WITH_CUDNN

}    // namespace dragon 

#endif    // DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_